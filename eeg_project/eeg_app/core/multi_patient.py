"""Traitement multi-patients : chargement, features, augmentation, fusion."""
import os

import pandas as pd
import streamlit as st

from eeg_app.config import (AUG_JITTER_MAX_ECH, AUG_SIGMA_MAX, AUG_SNR_MIN_DB,
                            DOSSIER_DATA, AUG_SEED)
from eeg_app.core.annotations import aligner_crises, extraire_crises
from eeg_app.core.augmentation import METHODES, augmenter_fenetres_crise
from eeg_app.core.data_loader import (charger_annotations, charger_donnees,
                                      charger_journal, chemin_journal,
                                      identifiants_patient)
from eeg_app.core.features import construire_dataset
from eeg_app.core.spectral import detecter_pic


@st.cache_data(show_spinner="Construction du dataset multi-patients...")
def construire_dataset_patient(fichier, chemin_annot, canal_analyse,
                               window_ms, overlap, seuil_crise, denses,
                               marge_art, exclure_art):
    """Construit le dataset d'un enregistrement (features + augmentation)."""
    if not chemin_annot or not os.path.isfile(chemin_annot):
        return None, "Fichier d'annotations introuvable"

    patient_id, id_annot = identifiants_patient(fichier)
    chemin_fichier = os.path.join(DOSSIER_DATA, fichier)
    df = charger_donnees(chemin_fichier)
    if canal_analyse not in df.columns:
        return None, f"{patient_id} : canal {canal_analyse} absent"
    fs = 1 / df["Time"].diff().median()
    if not pd.notna(fs) or fs <= 0:
        return None, f"{patient_id} : frequence d'echantillonnage invalide"

    # 1. Annotations
    annot = charger_annotations(chemin_annot)
    crises_brutes = extraire_crises(annot, id_annot)
    if crises_brutes is None or crises_brutes.empty:
        return None, f"{patient_id} : aucune crise annotee"

    # 2. Detection du pic
    t_spec, puiss, t_pic_auto, seuil = detecter_pic(chemin_fichier, canal_analyse, fs)

    # 3. Crise de reference (la 1ere) — pour l'alignement automatique
    ligne_ref = crises_brutes.iloc[0]
    t_min = float(df["Time"].iloc[0])
    t_max = float(df["Time"].iloc[-1])
    crises_toutes, crises, intervalles = aligner_crises(
        crises_brutes, ligne_ref, t_pic_auto, t_min, t_max)

    if crises.empty:
        return None, f"{patient_id} : aucune crise dans l'enregistrement"

    # 4. Construction du dataset de fenetres
    try:
        ds = construire_dataset(chemin_fichier, canal_analyse, fs,
                                window_ms, overlap, intervalles,
                                seuil_crise=seuil_crise, densifier=denses,
                                normaliser=True)
    except ValueError as e:
        return None, f"{patient_id} : {e}"
    if ds is None:
        return None, f"{patient_id} : signal trop court"

    # 5. Journal + artefacts
    chemin_j = chemin_journal(DOSSIER_DATA, fichier)
    journal = charger_journal(chemin_j)
    if journal is not None:
        from eeg_app.core.annotations import marquer_artefacts
        ds["Artefact"] = marquer_artefacts(ds, journal, marge_art)
        if exclure_art:
            ds = ds[~ds["Artefact"]].reset_index(drop=True)

    # 7. Metadonnees
    ds["Patient_ID"] = id_annot
    ds["Recording_ID"] = patient_id
    aug = None
    if (ds["Class"] == 1).any():
        params = {
            "max_shift": min(5, AUG_JITTER_MAX_ECH),
            "snr_db": AUG_SNR_MIN_DB + 5,
            "sigma_gain": min(0.05, AUG_SIGMA_MAX),
            "sigma_warp": min(0.05, AUG_SIGMA_MAX),
        }
        aug, _ = augmenter_fenetres_crise(
            chemin_fichier, canal_analyse, fs, window_ms, ds,
            list(METHODES), copies=2, params=params, seed=AUG_SEED,
            normaliser=True)
    if aug is not None and len(aug):
        aug["Patient_ID"] = id_annot
        aug["Recording_ID"] = patient_id

    return (ds, aug), None


def construire_dataset_multi_patients(fichiers, chemin_annot, canal_analyse,
                                      window_ms, overlap, seuil_crise, denses,
                                      marge_art, exclure_art):
    """Traite tous les patients et retourne un dataset fusionne.

    Retourne (dataset, aug, erreurs) ou aug est le pool des augmentations.
    """
    datasets, augs, erreurs = [], [], []
    for f in fichiers:
        try:
            resultat, err = construire_dataset_patient(
                f, chemin_annot, canal_analyse, window_ms, overlap,
                seuil_crise, denses, marge_art, exclure_art)
        except (KeyError, OSError, ValueError) as e:
            erreurs.append(f"{f} : {e}")
            continue
        if resultat is None:
            erreurs.append(err)
            continue
        ds, aug = resultat
        datasets.append(ds)
        if aug is not None and len(aug):
            augs.append(aug)

    if not datasets:
        return None, None, erreurs

    dataset = pd.concat(datasets, ignore_index=True)
    aug = pd.concat(augs, ignore_index=True) if augs else None
    return dataset, aug, erreurs