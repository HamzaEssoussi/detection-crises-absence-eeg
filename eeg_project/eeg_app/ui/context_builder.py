"""Construit l'AppContext : lit les widgets de la sidebar et appelle la logique metier."""
import os

import streamlit as st

from eeg_app.config import DOSSIER_DATA
from eeg_app.core.annotations import aligner_crises, extraire_crises
from eeg_app.core.context import AppContext
from eeg_app.core.data_loader import (charger_annotations, charger_donnees, charger_journal,
                                      chemin_journal, identifiants_patient,
                                      lister_enregistrements, trouver_annotations)
from eeg_app.core.spectral import detecter_pic
from eeg_app.ui import sidebar


def construire_contexte():
    # --- Verifications du dossier de donnees ---
    if not os.path.exists(DOSSIER_DATA):
        st.error(f"Le dossier '{DOSSIER_DATA}' n'existe pas. Lancez d'abord le script de conversion.")
        st.stop()

    fichiers = lister_enregistrements(DOSSIER_DATA)
    if not fichiers:
        st.error("Aucun fichier CSV trouve.")
        st.stop()

    # --- Enregistrement ---
    fichier = sidebar.choisir_enregistrement(fichiers)
    patient_id, id_annot = identifiants_patient(fichier)
    chemin_fichier = os.path.join(DOSSIER_DATA, fichier)
    df = charger_donnees(chemin_fichier)
    fs = 1 / df["Time"].diff().median()
    duree_totale = df["Time"].iloc[-1] - df["Time"].iloc[0]
    canaux_eeg = [c for c in df.columns if c.startswith("EEG")]

    # --- Parametres sidebar ---
    canaux_sel, debut, duree = sidebar.parametres_affichage(df, canaux_eeg)
    canal_analyse, window_sec, overlap, seuil_crise, densifier = sidebar.parametres_analyse(canaux_eeg)
    chemin_annot = sidebar.champ_annotations(trouver_annotations(DOSSIER_DATA) or "")

    ctx = AppContext(
        fichier=fichier, patient_id=patient_id, id_annot=id_annot,
        chemin_fichier=chemin_fichier, df=df, fs=fs, duree_totale=duree_totale,
        canaux_eeg=canaux_eeg, canaux_selectionnes=canaux_sel, debut=debut, duree=duree,
        canal_analyse=canal_analyse, window_sec=window_sec, overlap=overlap,
        seuil_crise=seuil_crise, densifier=densifier,
        chemin_annot=chemin_annot,
    )

    # --- Annotations + alignement ---
    if chemin_annot and os.path.exists(chemin_annot):
        try:
            ctx.annot = charger_annotations(chemin_annot)
            ctx.crises_brutes = extraire_crises(ctx.annot, id_annot)
        except Exception as e:
            ctx.erreur_annot = str(e)

    if ctx.crises_brutes is not None and not ctx.crises_brutes.empty:
        ctx.t_spec, ctx.puiss, t_pic_auto, ctx.seuil = detecter_pic(chemin_fichier, canal_analyse, fs)
        debuts_trace = ctx.crises_brutes.get("DebutTrace_brut")
        if debuts_trace is not None and debuts_trace.notna().any():
            ctx.t_pic = t_pic_auto
            ligne_ref = ctx.crises_brutes.iloc[0]
            st.sidebar.caption("Alignement des crises a partir de DebutTrace.")
        else:
            ctx.t_pic, ligne_ref = sidebar.parametres_pic(
                fichier, canal_analyse, t_pic_auto, duree_totale, ctx.crises_brutes)
        t_min, t_max = float(df["Time"].iloc[0]), float(df["Time"].iloc[-1])
        ctx.crises_toutes, ctx.crises, ctx.intervalles = aligner_crises(
            ctx.crises_brutes, ligne_ref, ctx.t_pic, t_min, t_max)

    # --- Journal du technicien ---
    chemin_j = chemin_journal(DOSSIER_DATA, fichier)
    ctx.journal = charger_journal(chemin_j)
    ctx.exclure_art, ctx.marge_art = sidebar.parametres_journal(ctx.journal, chemin_j)

    return ctx

def construire_contexte_multi():
    """Construit un contexte multi-patients (pool de tous les patients)."""
    from eeg_app.core.multi_patient import construire_dataset_multi_patients
    from eeg_app.core.context import AppContext

    if not os.path.exists(DOSSIER_DATA):
        st.error(f"Le dossier '{DOSSIER_DATA}' n'existe pas.")
        st.stop()

    fichiers = lister_enregistrements(DOSSIER_DATA)
    if not fichiers:
        st.error("Aucun fichier CSV trouve.")
        st.stop()

    # Sidebar multi
    patients_choisis = sidebar.choisir_patients_multi(fichiers)
    if not patients_choisis:
        st.warning("Selectionnez au moins un enregistrement EEG.")
        st.stop()

    # Parametres communs
    chemin_annot = sidebar.champ_annotations(trouver_annotations(DOSSIER_DATA) or "")
    if not chemin_annot or not os.path.isfile(chemin_annot):
        st.error("Le fichier d'annotations n'existe pas. Indiquez un fichier valide dans la barre laterale.")
        st.stop()

    canaux_communs = None
    for fichier in patients_choisis:
        df_patient = charger_donnees(os.path.join(DOSSIER_DATA, fichier))
        canaux_patient = {c for c in df_patient.columns if c.startswith("EEG")}
        canaux_communs = canaux_patient if canaux_communs is None else canaux_communs & canaux_patient
    if not canaux_communs:
        st.error("Aucun canal EEG n'est commun aux enregistrements selectionnes.")
        st.stop()

    canal_analyse, window_sec, overlap, seuil_crise, denses = sidebar.parametres_analyse(
        sorted(canaux_communs))

    journaux = [
        charger_journal(chemin_journal(DOSSIER_DATA, fichier))
        for fichier in patients_choisis
    ]
    journaux_disponibles = [journal for journal in journaux if journal is not None]
    nb_notes = sum(len(journal) for journal in journaux_disponibles)
    exclure_art, marge_art = sidebar.parametres_journal_multi(
        len(journaux_disponibles), nb_notes)

    # Construire le dataset pool
    dataset, aug, erreurs = construire_dataset_multi_patients(
        patients_choisis, chemin_annot, canal_analyse,
        window_sec, overlap, seuil_crise, denses, marge_art, exclure_art)

    if dataset is None:
        st.error("Aucun patient n'a pu etre traite.")
        for e in erreurs:
            st.warning(e)
        st.stop()
    for erreur in erreurs:
        st.warning(erreur)

    # Construire un contexte "virtuel"
    ctx = AppContext(
        fichier=f"multi ({len(patients_choisis)} patients)",
        patient_id=f"{len(patients_choisis)} patients",
        id_annot="multi",
        chemin_fichier="",
        df=dataset,                # ← le dataset fusionne
        fs=0.0,
        duree_totale=0.0,
        canaux_eeg=[],
        canaux_selectionnes=[],
        debut=0, duree=0,
        canal_analyse=canal_analyse,
        window_sec=window_sec,
        overlap=overlap,
        seuil_crise=seuil_crise,
        densifier=denses,
        chemin_annot=chemin_annot,
        journal=None,
        exclure_art=exclure_art,
        marge_art=marge_art,
        mode="multi",
        dataset_preconstruit=dataset,
        augmentation=aug,
    )
    return ctx