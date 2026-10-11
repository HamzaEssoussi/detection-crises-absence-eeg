"""Chargement des fichiers (EEG, annotations Excel/CSV, journal du technicien)."""
import os
import re

import pandas as pd
import streamlit as st


# ---------- Enregistrements EEG ----------
def lister_enregistrements(dossier):
    """Fichiers de signal (suffixe 'd.csv') disponibles dans le dossier."""
    return sorted(f for f in os.listdir(dossier) if f.endswith("d.csv"))


def identifiants_patient(fichier):
    """Retourne (patient_id, id_annot) a partir du nom du fichier."""
    patient_id = fichier.replace("d.csv", "")
    id_annot = re.sub(r"_000[01]$", "", patient_id).replace("_", "-")
    return patient_id, id_annot


def chemin_journal(dossier, fichier):
    """Chemin du fichier journal ('h.csv') associe a un fichier de signal."""
    return os.path.join(dossier, fichier[:-len("d.csv")] + "h.csv")


@st.cache_data
def charger_donnees(chemin):
    return pd.read_csv(chemin, encoding="latin-1")


# ---------- Annotations ----------
def charger_annotations(chemin):
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier d'annotations introuvable : {chemin}")
    stat = os.stat(chemin)
    return _charger_annotations_cache(chemin, stat.st_mtime_ns, stat.st_size)


@st.cache_data
def _charger_annotations_cache(chemin, modification_ns, taille):
    del modification_ns, taille
    if chemin.lower().endswith(".csv"):
        return pd.read_csv(chemin)
    return pd.read_excel(chemin)


def trouver_annotations(dossier):
    for nom in ("Annotations.xlsx", "Annotations.csv"):
        for base in (dossier, os.path.dirname(dossier)):
            p = os.path.join(base, nom)
            if os.path.exists(p):
                return p
    return None


# ---------- Journal du technicien ----------
def classer_evenement(texte):
    t = str(texte).upper()
    if "ARTEFACT" in t or "BOUGE" in t or "SATURATION" in t:
        return "artefact"
    if "CRISE" in t or "ABS" in t:
        return "crise"
    if t.strip() in ("YO", "YF"):
        return "yeux"
    if t.startswith("HPN"):
        return "hpn"
    if t.startswith("SLI"):
        return "sli"
    return "autre"


@st.cache_data
def charger_journal(chemin):
    if not os.path.exists(chemin):
        return None
    try:
        j = pd.read_csv(chemin, encoding="utf-8", encoding_errors="replace")
        if "Onset" not in j.columns or "Annotations" not in j.columns:
            return None
        onset = pd.to_numeric(j["Onset"], errors="coerce")
        if onset.isna().all():
            onset = pd.to_timedelta(j["Onset"], errors="coerce").dt.total_seconds()
        j = pd.DataFrame({"Onset": onset, "Annotations": j["Annotations"].astype(str)})
        j = j.dropna(subset=["Onset"]).reset_index(drop=True)
        j["Type"] = j["Annotations"].apply(classer_evenement)
        return j
    except Exception:
        return None
