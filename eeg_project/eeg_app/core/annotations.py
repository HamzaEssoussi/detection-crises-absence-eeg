"""Traitement des annotations : extraction, alignement temporel, verifications."""
import re
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from eeg_app.config import DUREE_MAX_CRISE_S, SECONDES_PAR_JOUR


def _heure_en_fraction_jour(valeur, libelle):
    """Convertit une fraction Excel ou une heure lisible en fraction de jour."""
    if pd.isna(valeur):
        return None

    if isinstance(valeur, (datetime, time, timedelta, pd.Timestamp, pd.Timedelta)):
        secondes = (
            valeur.hour * 3600 + valeur.minute * 60 + valeur.second
            + getattr(valeur, "microsecond", 0) / 1_000_000
            if hasattr(valeur, "hour") else valeur.total_seconds()
        )
        return (secondes % SECONDES_PAR_JOUR) / SECONDES_PAR_JOUR

    texte = str(valeur).strip()
    try:
        numerique = float(texte.replace(",", "."))
    except ValueError:
        numerique = None
    if numerique is not None:
        if 0 <= numerique < 1:
            return numerique
        raise ValueError(
            f"{libelle} invalide : {valeur!r} (fraction de jour attendue entre 0 et 1).")

    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?", texte)
    if match is None:
        raise ValueError(
            f"{libelle} invalide : {valeur!r} (formats acceptes : fraction Excel ou HH:MM:SS).")
    heure, minute, seconde, fraction = match.groups()
    heure, minute = int(heure), int(minute)
    seconde = int(seconde or 0)
    microsecondes = int(((fraction or "") + "000000")[:6])
    if heure > 23 or minute > 59 or seconde > 59:
        raise ValueError(f"{libelle} invalide : {valeur!r}.")
    total = heure * 3600 + minute * 60 + seconde + microsecondes / 1_000_000
    return total / SECONDES_PAR_JOUR


def extraire_crises(annot, id_patient):
    """Extrait les crises de toutes les lignes portant le meme ID."""
    id_col = next(
        (c for c in annot.columns
         if re.sub(r"[^a-z]", "", str(c).casefold())
         in {"id", "patientid", "idpatient", "identifiant", "codepatient"}),
        annot.columns[0],
    )
    ids = annot[id_col].astype(str).str.strip()
    lignes = annot[ids == id_patient]
    if lignes.empty:
        return None

    debut_trace_col = next(
        (c for c in annot.columns
         if re.sub(r"[^a-z]", "", str(c).casefold()) == "debuttrace"),
        None,
    )
    paires = set()
    for i in range(1, 13):
        col_deb = next((c for c in annot.columns
                        if re.fullmatch(
                            rf"(Deb|Debut|Debt)CE{i}", str(c).strip(), re.IGNORECASE)), None)
        col_fin = next((c for c in annot.columns
                        if re.fullmatch(rf"FinCE{i}", str(c).strip(), re.IGNORECASE)), None)
        if col_deb is None or col_fin is None:
            continue
        for numero_ligne, ligne in lignes.iterrows():
            brut_deb, brut_fin = ligne[col_deb], ligne[col_fin]
            deb_present = pd.notna(brut_deb) and str(brut_deb).strip() != ""
            fin_present = pd.notna(brut_fin) and str(brut_fin).strip() != ""
            if not deb_present and not fin_present:
                continue
            if deb_present != fin_present:
                raise ValueError(
                    f"{id_patient} : debut ou fin manquant pour la crise {i} "
                    f"(ligne d'annotation {numero_ligne + 2}).")

            deb = _heure_en_fraction_jour(
                brut_deb, f"{id_patient}, debut crise {i}, ligne {numero_ligne + 2}")
            fin = _heure_en_fraction_jour(
                brut_fin, f"{id_patient}, fin crise {i}, ligne {numero_ligne + 2}")
            fin_corrigee = float(fin)
            if fin_corrigee <= deb:
                fin_corrigee += 1.0
            duree_s = (fin_corrigee - deb) * SECONDES_PAR_JOUR
            if duree_s > DUREE_MAX_CRISE_S:
                raise ValueError(
                    f"{id_patient} : crise {i} de {duree_s / 3600:.2f} h "
                    f"(maximum autorise : {DUREE_MAX_CRISE_S} s).")
            debut_trace = None
            if debut_trace_col is not None:
                valeur_trace = ligne[debut_trace_col]
                if pd.notna(valeur_trace) and str(valeur_trace).strip():
                    debut_trace = _heure_en_fraction_jour(
                        valeur_trace,
                        f"{id_patient}, DebutTrace, ligne {numero_ligne + 2}")
            paires.add((deb, fin_corrigee, debut_trace))

    crises = [
        {"Crise": numero, "Debut_brut": deb, "Fin_brut": fin,
         "DebutTrace_brut": debut_trace}
        for numero, (deb, fin, debut_trace) in enumerate(
            sorted(paires, key=lambda paire: (paire[0], paire[1], paire[2] or -1)),
            start=1)
    ]
    return pd.DataFrame(
        crises, columns=["Crise", "Debut_brut", "Fin_brut", "DebutTrace_brut"])


def aligner_crises(crises_brutes, ligne_ref, t_pic, t_min, t_max):
    """Convertit les crises en secondes relatives au debut de l'EEG.

    Retourne (crises_toutes, crises_dans_fichier, intervalles).
    """
    crises_toutes = crises_brutes.copy()
    reference_offset = ligne_ref["Debut_brut"] * SECONDES_PAR_JOUR - t_pic
    debuts, fins = [], []
    for _, crise in crises_toutes.iterrows():
        trace = crise.get("DebutTrace_brut")
        if pd.notna(trace):
            debut_s = (crise["Debut_brut"] - trace) * SECONDES_PAR_JOUR
            if debut_s < -SECONDES_PAR_JOUR / 2:
                debut_s += SECONDES_PAR_JOUR
            fin_s = (crise["Fin_brut"] - trace) * SECONDES_PAR_JOUR
            while fin_s < debut_s:
                fin_s += SECONDES_PAR_JOUR
            debuts.append(t_min + debut_s)
            fins.append(t_min + fin_s)
        else:
            debuts.append(crise["Debut_brut"] * SECONDES_PAR_JOUR - reference_offset)
            fins.append(crise["Fin_brut"] * SECONDES_PAR_JOUR - reference_offset)
    crises_toutes["Debut_s"] = debuts
    crises_toutes["Fin_s"] = fins
    crises_toutes["Duree_s"] = crises_toutes["Fin_s"] - crises_toutes["Debut_s"]

    dans_fichier = (crises_toutes["Fin_s"] >= t_min) & (crises_toutes["Debut_s"] <= t_max)
    crises = crises_toutes[dans_fichier].reset_index(drop=True)
    intervalles = tuple((round(float(a), 3), round(float(b), 3))
                        for a, b in zip(crises["Debut_s"], crises["Fin_s"]))
    return crises_toutes, crises, intervalles


def marquer_artefacts(dataset, journal, marge):
    """Vecteur booleen : la fenetre touche-t-elle un artefact du journal ?"""
    ons = np.sort(journal.loc[journal["Type"] == "artefact", "Onset"].values)
    if len(ons) == 0:
        return np.zeros(len(dataset), dtype=bool)
    debut_f = dataset["Window_start_s"].values - marge
    fin_f = dataset["Window_end_s"].values + marge
    return np.searchsorted(ons, fin_f, side="right") > np.searchsorted(ons, debut_f, side="left")


def verifier_avec_journal(crises, journal):
    """Croise les crises annotees avec le journal du technicien.

    Retourne (tableau_verification, onsets_artefacts, onsets_notes_crise).
    """
    art = np.sort(journal.loc[journal["Type"] == "artefact", "Onset"].values)
    notes = np.sort(journal.loc[journal["Type"] == "crise", "Onset"].values)
    verif = crises[["Crise", "Debut_s", "Fin_s"]].copy()
    verif["Notes crise/absence (+/-30 s)"] = [
        int(((notes >= d - 30) & (notes <= f + 30)).sum())
        for d, f in zip(crises["Debut_s"], crises["Fin_s"])]
    verif["Artefacts pendant la crise"] = [
        int(((art >= d) & (art <= f)).sum())
        for d, f in zip(crises["Debut_s"], crises["Fin_s"])]
    return verif, art, notes


def repartition_cible(df, intervalles):
    """Nombre d'echantillons hors crise (0) et en crise (1)."""
    y_sample = np.zeros(len(df), dtype=int)
    for deb, fin in intervalles:
        y_sample[((df["Time"] >= deb) & (df["Time"] <= fin)).values] = 1
    return pd.Series(y_sample).value_counts().reindex([0, 1], fill_value=0)
