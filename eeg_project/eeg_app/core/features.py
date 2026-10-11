"""Segmentation en fenetres et extraction des features (temps + frequence)."""
import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft, rfftfreq

from eeg_app.config import BANDS, FEATURE_COLS, TAILLE_BLOC
from eeg_app.core.data_loader import charger_donnees


def features_fenetres(W, fs):
    freqs = rfftfreq(W.shape[1], d=1 / fs)
    P = np.abs(rfft(W, axis=1)) ** 2
    out = {
        "Mean": W.mean(axis=1),
        "Std": W.std(axis=1),
        "Variance": W.var(axis=1),
        "Min": W.min(axis=1),
        "Max": W.max(axis=1),
    }
    out["Amplitude"] = out["Max"] - out["Min"]
    out["RMS"] = np.sqrt((W ** 2).mean(axis=1))
    out["Energy"] = (W ** 2).sum(axis=1)
    for nom, (lo, hi) in BANDS.items():
        out[f"Power_{nom}"] = P[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
    return out


def normaliser_signal(signal, canal):
    """Standardise un canal sur l'enregistrement auquel il appartient."""
    x = np.asarray(signal, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"Le canal {canal} contient des valeurs non finies.")
    ecart_type = float(x.std())
    if not np.isfinite(ecart_type) or ecart_type == 0:
        raise ValueError(f"Le canal {canal} a une variance invalide et ne peut pas etre normalise.")
    return (x - x.mean()) / ecart_type


@st.cache_data(show_spinner="Construction du dataset de fenetres...")
def construire_dataset(chemin, canal, fs, window_sec, overlap, intervalles,
                       seuil_crise=0.3, densifier=True, normaliser=False):
    df = charger_donnees(chemin)
    t = df["Time"].values
    x = df[canal].values.astype(float)
    if normaliser:
        x = normaliser_signal(x, canal)

    y_sample = np.zeros(len(t))
    for deb, fin in intervalles:
        y_sample[(t >= deb) & (t <= fin)] = 1.0

    win = int(window_sec * fs)
    step = max(1, int(win * (1 - overlap)))
    if len(x) < win:
        return None

    starts = np.arange(0, len(x) - win + 1, step)
    if densifier and intervalles:
        pas_dense = max(1, win // 8)
        starts_denses = []
        max_start = len(x) - win
        for deb, fin in intervalles:
            premier = max(0, np.searchsorted(t, deb, side="left") - win + 1)
            dernier = min(max_start, np.searchsorted(t, fin, side="right") - 1)
            if premier <= dernier:
                denses = np.arange(premier, dernier + 1, pas_dense)
                if denses[-1] != dernier:
                    denses = np.append(denses, dernier)
                starts_denses.append(denses)
        if starts_denses:
            starts = np.unique(np.concatenate([starts, *starts_denses]))

    blocs = []
    classes = []
    fenetres = sliding_window_view(x, win)
    etiquettes = sliding_window_view(y_sample, win)
    for s in range(0, len(starts), TAILLE_BLOC):
        indices = starts[s:s + TAILLE_BLOC]
        blocs.append(pd.DataFrame(features_fenetres(fenetres[indices], fs)))
        classes.append((etiquettes[indices].mean(axis=1) >= seuil_crise).astype(int))
    dataset = pd.concat(blocs, ignore_index=True)

    dataset.insert(0, "Window", np.arange(1, len(dataset) + 1))
    dataset["Class"] = np.concatenate(classes)
    dataset["Window_start_s"] = t[starts]
    dataset["Window_end_s"] = t[starts + win - 1]
    return dataset


def moyenne_par_classe(dataset):
    """Moyenne de chaque feature pour la classe 0 (normal) et 1 (crise)."""
    comp = dataset.groupby("Class")[FEATURE_COLS].mean().T
    comp.columns = [f"Classe {c} ({'crise' if c == 1 else 'pas crise'})" for c in comp.columns]
    return comp
