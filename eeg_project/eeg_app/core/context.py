"""Contexte de l'application : regroupe l'etat partage entre les etapes.

Remplace les nombreuses variables globales du script d'origine.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from eeg_app.core.annotations import marquer_artefacts
from eeg_app.core.features import construire_dataset


@dataclass
class AppContext:
    # --- Enregistrement ---
    fichier: str
    patient_id: str
    id_annot: str
    chemin_fichier: str
    df: pd.DataFrame
    fs: float
    duree_totale: float
    canaux_eeg: list
    # --- Affichage (sidebar) ---
    canaux_selectionnes: list
    debut: int
    duree: int
    # --- Analyse (sidebar) ---
    canal_analyse: str
    window_sec: float
    overlap: float
    seuil_crise: float = 0.3
    densifier: bool = True
    # --- Annotations ---
    chemin_annot: str = ""
    annot: Optional[pd.DataFrame] = None
    erreur_annot: Optional[str] = None
    crises_brutes: Optional[pd.DataFrame] = None
    crises_toutes: Optional[pd.DataFrame] = None
    crises: Optional[pd.DataFrame] = None
    intervalles: tuple = ()
    # --- Detection du pic ---
    t_spec: Optional[object] = None
    puiss: Optional[object] = None
    seuil: Optional[float] = None
    t_pic: Optional[float] = None
    # --- Journal du technicien ---
    journal: Optional[pd.DataFrame] = None
    exclure_art: bool = False
    marge_art: float = 1.0
    mode: str = "mono"
    dataset_preconstruit: Optional[pd.DataFrame] = None
    augmentation: Optional[pd.DataFrame] = None

    def get_dataset_brut(self):
        """Dataset de fenetres, artefacts marques mais non exclus."""
        if self.dataset_preconstruit is not None:
            return self.dataset_preconstruit.copy()
        if self.crises is None:
            return None
        ds = construire_dataset(self.chemin_fichier, self.canal_analyse, self.fs,
                                self.window_sec, self.overlap, self.intervalles,
                                self.seuil_crise, self.densifier)
        if ds is None:
            return None
        ds = ds.copy()
        if self.journal is not None:
            ds["Artefact"] = marquer_artefacts(ds, self.journal, self.marge_art)
        return ds

    def get_dataset(self):
        """Dataset final (artefacts exclus si demande)."""
        ds = self.get_dataset_brut()
        if ds is not None and self.exclure_art and "Artefact" in ds.columns:
            ds = ds[~ds["Artefact"]].reset_index(drop=True)
        return ds