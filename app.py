import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
st.set_page_config(page_title="Détection Crises EEG", page_icon="🧠", layout="wide")
st.title("Détection Automatique des Crises d'Absence")
st.markdown("---")

# Chemin vers les CSV
DOSSIER_DATA = r"C:\Users\LENOVO\Desktop\Projet_EEG_DataMining\Projet data mining 3AINFO\data_csv"

@st.cache_data
def charger_donnees(chemin):
    return pd.read_csv(chemin, encoding='latin-1')

# Lister les fichiers disponibles
if not os.path.exists(DOSSIER_DATA):
    st.error(f"Le dossier '{DOSSIER_DATA}' n'existe pas. Lancez d'abord le script de conversion.")
    st.stop()

fichiers_disponibles = [f for f in os.listdir(DOSSIER_DATA) if f.endswith('d.csv')]

if not fichiers_disponibles:
    st.error("Aucun fichier CSV trouvé.")
    st.stop()

# Sélection du patient
st.sidebar.header("⚙️ Paramètres")
fichier_selectionne = st.sidebar.selectbox("Choisir un enregistrement", fichiers_disponibles)
patient_id = fichier_selectionne.replace('d.csv', '')

# Chargement
chemin_fichier = os.path.join(DOSSIER_DATA, fichier_selectionne)
df = charger_donnees(chemin_fichier)

# Informations générales
st.subheader(f"📊 Patient : {patient_id}")

dt = df['Time'].iloc[1] - df['Time'].iloc[0]
fs = 1 / dt

col1, col2, col3, col4 = st.columns(4)
col1.metric("Lignes", f"{df.shape[0]:,}")
col2.metric("Colonnes", df.shape[1])
col3.metric("Fréquence (Hz)", f"{fs:.2f}")
col4.metric("Durée (s)", f"{df['Time'].iloc[-1]:.1f}")

# Sélection des canaux et de la durée
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Affichage")

canaux_eeg = [col for col in df.columns if col.startswith('EEG')]
canaux_selectionnes = st.sidebar.multiselect("Canaux à afficher", canaux_eeg, default=canaux_eeg[:4])

duree_max = int(df['Time'].iloc[-1])
debut = st.sidebar.slider("Début (secondes)", 0, duree_max - 1, 0)
duree = st.sidebar.slider("Durée (secondes)", 1, 30, 5)

# Graphique
st.subheader(f"📈 Signal EEG - {debut}s à {debut + duree}s")

if canaux_selectionnes:
    debut_idx = int(debut * fs)
    fin_idx = int((debut + duree) * fs)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, canal in enumerate(canaux_selectionnes):
        signal = df[canal].iloc[debut_idx:fin_idx].values
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        ax.plot(df['Time'].iloc[debut_idx:fin_idx], signal_norm + i * 3, label=canal, linewidth=0.8)
    
    ax.set_xlabel("Temps (secondes)")
    ax.set_ylabel("Amplitude (normalisée + décalage)")
    ax.set_yticks([])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    st.pyplot(fig)
else:
    st.warning("Sélectionnez au moins un canal.")

# Statistiques
st.markdown("---")
st.subheader("📉 Statistiques descriptives")
if canaux_selectionnes:
    stats = df[canaux_selectionnes].describe().T
    st.dataframe(stats, use_container_width=True)

st.caption("Mini-projet Data Mining - Détection des crises d'absence EEG")