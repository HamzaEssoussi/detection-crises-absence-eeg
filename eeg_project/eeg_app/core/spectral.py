"""Analyse spectrale : detection du pic de crise (pointes-ondes ~3 Hz)."""
import numpy as np
import streamlit as st
from scipy.signal import spectrogram

from eeg_app.config import BANDE_PIC
from eeg_app.core.data_loader import charger_donnees


@st.cache_data(show_spinner="Detection du pic spectral...")
def detecter_pic(chemin, canal, fs):
    df = charger_donnees(chemin)
    signal = df[canal].values
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=int(fs), noverlap=int(fs * 0.5))
    mask = (f >= BANDE_PIC[0]) & (f <= BANDE_PIC[1])
    puissance = Sxx[mask, :].sum(axis=0)
    seuil = puissance.mean() + 3 * puissance.std()
    idx = int(np.argmax(puissance))
    niveau = max(seuil, 0.5 * puissance[idx])
    debut_idx = idx
    while debut_idx > 0 and puissance[debut_idx - 1] >= niveau:
        debut_idx -= 1
    t_debut = t[debut_idx] - 0.5 * int(fs) / fs
    t0 = df["Time"].iloc[0]
    return t + t0, puissance, float(max(t_debut, 0.0) + t0), float(seuil)
