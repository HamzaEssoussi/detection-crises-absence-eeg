# ============================================================
# Detection automatique des crises d'absence - EEG (Streamlit)
# Lancer :  streamlit run app.py
# Dependances : streamlit pandas numpy scipy matplotlib seaborn
#               scikit-learn openpyxl
#
# Arborescence attendue :
#   app.py
#   image/
#       banner_hero.png
#       fonctions.png
#       composants.png
#       processus.png
#       faits.png
# ============================================================
import os
import re
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------
# Configuration generale
# ------------------------------------------------------------
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")

st.set_page_config(
    page_title="Detection Crises EEG",
    layout="wide",
    initial_sidebar_state="expanded",   # <-- sidebar ouverte par defaut
)

DOSSIER_DATA = os.environ.get(
    "EEG_DATA_DIR",
    r"C:\Users\LENOVO\Desktop\Projet_EEG_DataMining\Projet data mining 3AINFO\data_csv",
)
SECONDES_PAR_JOUR = 86400
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
FEATURE_COLS = ["Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy",
                "Power_Delta", "Power_Theta", "Power_Alpha", "Power_Beta"]

STEPS = ["Signal", "Crises & cible", "Features", "Modeles"]


# ------------------------------------------------------------
# Chargement de l'image de fond (encodage base64)
# ------------------------------------------------------------
def image_base64(nom_fichier):
    chemin = os.path.join(IMAGE_DIR, nom_fichier)
    if not os.path.exists(chemin):
        return None
    with open(chemin, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ------------------------------------------------------------
# Theme visuel SOMBRE avec image de fond et sidebar collapsible
# ------------------------------------------------------------
def charger_theme():
    bg_b64 = image_base64("banner_hero.png")
    style_bg = ""
    if bg_b64:
        style_bg = f"""
        .stApp {{
            background-image: 
                linear-gradient(rgba(20, 20, 28, 0.88), rgba(15, 15, 22, 0.92)),
                url('data:image/png;base64,{bg_b64}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        :root {{
            --bg-main: #1a1a24;
            --bg-panel: #23232e;
            --bg-panel-alt: #2a2a38;
            --border-color: #3a3a48;
            --accent-blue: #5b9cff;
            --accent-blue-dark: #2f5df5;
            --accent-blue-soft: rgba(91, 156, 255, 0.18);
            --accent-red: #ff5252;
            --accent-red-soft: rgba(255, 82, 82, 0.15);
            --accent-orange: #ffa726;
            --accent-green: #4ade80;
            --text-primary: #ffffff;
            --text-muted: #c5c5d0;
            --text-light: #9a9aab;
        }}

        /* ========== SUPPRESSION DE LA BARRE BLANCHE STREAMLIT ========== */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0px !important;
            min-height: 0px !important;
            padding: 0 !important;
            display: none !important;
        }}
        div[data-testid="stToolbar"] {{ display: none !important; }}
        div[data-testid="stDecoration"] {{ display: none !important; }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        .main .block-container {{
            padding-top: 0.5rem !important;
            padding-bottom: 1rem !important;
            max-width: 100% !important;
        }}

        /* ========== FOND GENERAL ========== */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }}

        {style_bg}

        .stApp {{
            color: #ffffff;
            background-color: var(--bg-main);
        }}

        html, body, p, span, div, label, li {{
            color: #ffffff;
        }}

        /* ========== SIDEBAR (BARRE A GAUCHE) ========== */
        section[data-testid="stSidebar"] {{
            background-color: #1e1e28 !important;
            border-right: 1px solid var(--border-color);
        }}

        section[data-testid="stSidebar"] > div {{
            background-color: #1e1e28 !important;
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--text-muted) !important;
        }}

        section[data-testid="stSidebar"] label {{
            color: var(--text-muted) !important;
            font-weight: 500 !important;
        }}

        /* ---- Bouton pour cacher/afficher la sidebar ---- */
        button[data-testid="stBaseButton-headerNoPadding"],
        button[kind="header"],
        button[kind="headerNoPadding"] {{
            background-color: #2f5df5 !important;
            color: #ffffff !important;
            border-radius: 50% !important;
            border: none !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 2px 8px rgba(47, 93, 245, 0.5);
            z-index: 999999 !important;
        }}

        button[data-testid="stBaseButton-headerNoPadding"]:hover,
        button[kind="header"]:hover,
        button[kind="headerNoPadding"]:hover {{
            background-color: #1c3fd1 !important;
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(47, 93, 245, 0.7);
        }}

        button[data-testid="stBaseButton-headerNoPadding"] svg,
        button[kind="header"] svg,
        button[kind="headerNoPadding"] svg {{
            fill: #ffffff !important;
            color: #ffffff !important;
            width: 16px !important;
            height: 16px !important;
        }}

        /* Bouton quand la sidebar est fermee */
        [data-testid="stSidebarCollapsedControl"] {{
            background-color: #2f5df5 !important;
            color: #ffffff !important;
            border-radius: 0 8px 8px 0 !important;
            box-shadow: 0 2px 8px rgba(47, 93, 245, 0.5);
            z-index: 999999 !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}

        [data-testid="stSidebarCollapsedControl"]:hover {{
            background-color: #1c3fd1 !important;
            box-shadow: 0 4px 12px rgba(47, 93, 245, 0.7);
        }}

        [data-testid="stSidebarCollapsedControl"] svg {{
            fill: #ffffff !important;
            color: #ffffff !important;
        }}

        /* Widgets dans la sidebar */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: #2a2a38 !important;
            border-color: var(--border-color) !important;
            color: var(--text-muted) !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
            color: var(--text-muted) !important;
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {{
            background-color: #23232e !important;
            color: var(--text-muted) !important;
        }}
        div[data-baseweb="popover"] li:hover {{
            background-color: var(--accent-blue-dark) !important;
        }}

        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {{
            background-color: #2a2a38 !important;
            color: var(--text-muted) !important;
            border: 1px solid var(--border-color) !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {{
            background-color: var(--accent-blue) !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stSliderThumbValue"],
        section[data-testid="stSidebar"] div[data-testid="stSliderTickBarMin"],
        section[data-testid="stSidebar"] div[data-testid="stSliderTickBarMax"] {{
            color: var(--text-muted) !important;
        }}

        section[data-testid="stSidebar"] input[type="checkbox"] {{
            accent-color: var(--accent-blue) !important;
        }}

        section[data-testid="stSidebar"] hr {{
            border-color: var(--border-color) !important;
        }}

        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] small {{
            color: var(--text-muted) !important;
        }}

        section[data-testid="stSidebar"] .section-title {{
            color: var(--accent-blue) !important;
            font-size: 0.95rem !important;
            border-bottom: 2px solid var(--accent-blue-soft);
            padding-bottom: 4px;
            margin-bottom: 10px;
        }}

        /* ========== TITRES ========== */
        h1, h2, h3, h4, .app-title, .section-title {{
            font-family: 'Oswald', sans-serif;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #ffffff !important;
        }}

        /* ========== BARRE D'ETAPES ========== */
        div[data-testid="column"] .step-btn button {{
            width: 100%;
            border-radius: 8px !important;
            border: 1.5px solid var(--border-color) !important;
            background-color: rgba(35, 35, 46, 0.9) !important;
            color: #ffffff !important;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.82rem;
            padding: 12px 6px !important;
            transition: all 0.2s ease;
        }}
        div[data-testid="column"] .step-btn button:hover {{
            border-color: var(--accent-blue) !important;
            color: #ffffff !important;
            background-color: var(--accent-blue-soft) !important;
        }}
        div[data-testid="column"] .step-btn-active button {{
            background-color: var(--accent-blue-dark) !important;
            border-color: var(--accent-blue-dark) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(91, 156, 255, 0.5);
        }}

        /* ========== BANDEAU DE TITRE ========== */
        .app-header {{
            position: relative;
            padding: 36px 40px;
            margin-bottom: 20px;
            margin-top: 8px;
            background: linear-gradient(120deg, rgba(35, 35, 46, 0.92) 0%, rgba(26, 26, 36, 0.88) 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}
        .app-header .eyebrow {{
            color: var(--accent-blue);
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }}
        .app-title {{
            font-size: 2.4rem;
            line-height: 1.05;
            margin: 8px 0 14px 0;
            color: #ffffff !important;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.6);
        }}
        .app-subtitle {{
            display: inline-block;
            background-color: var(--accent-blue-dark);
            color: #ffffff !important;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.92rem;
            padding: 8px 18px;
            border-radius: 999px;
        }}

        /* ========== SECTIONS ========== */
        .section-block {{
            margin-top: 26px;
            margin-bottom: 14px;
        }}
        .section-title {{
            font-size: 1.35rem;
            margin-bottom: 6px;
            color: #ffffff !important;
        }}
        .section-bar {{
            width: 56px;
            height: 4px;
            background-color: var(--accent-blue);
            border-radius: 2px;
            margin-bottom: 10px;
        }}
        .section-desc {{
            color: var(--text-muted) !important;
            font-size: 0.92rem;
            max-width: 780px;
        }}

        /* ========== CALLOUTS ========== */
        .callout {{
            background-color: rgba(91, 156, 255, 0.15) !important;
            border: 1px solid rgba(91, 156, 255, 0.4);
            border-left: 4px solid var(--accent-blue);
            border-radius: 8px;
            padding: 16px 20px;
            font-size: 0.9rem;
            color: #ffffff !important;
            margin: 10px 0 18px 0;
        }}
        .callout * {{ color: #ffffff !important; }}
        .callout b {{ color: var(--accent-blue) !important; }}
        .callout ul {{ margin: 6px 0 0 0; padding-left: 18px; }}
        .callout li {{ margin-bottom: 4px; }}

        .callout-warning {{
            background-color: rgba(255, 82, 82, 0.15) !important;
            border: 1px solid rgba(255, 82, 82, 0.4);
            border-left: 4px solid var(--accent-red);
        }}
        .callout-warning b {{ color: var(--accent-red) !important; }}

        /* ========== LEGENDES ========== */
        .legend-row {{ display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px; }}
        .swatch {{ min-width: 12px; height: 12px; border-radius: 3px; margin-top: 4px; }}
        .legend-text {{ color: var(--text-muted) !important; font-size: 0.9rem; }}
        .legend-text b {{ color: #ffffff !important; }}

        /* ========== METRICS ========== */
        div[data-testid="stMetric"] {{
            background-color: rgba(35, 35, 46, 0.9) !important;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px 16px 8px 16px;
        }}
        div[data-testid="stMetricLabel"] {{
            color: var(--text-muted) !important;
            text-transform: uppercase;
            font-size: 0.72rem !important;
            letter-spacing: 0.06em;
        }}
        div[data-testid="stMetricValue"] {{
            color: #ffffff !important;
            font-weight: 600;
        }}

        /* ========== BOUTONS ========== */
        div[data-testid="stButton"] > button {{
            border-radius: 8px;
            font-weight: 600;
            color: #ffffff !important;
            border-color: var(--border-color) !important;
            background-color: rgba(35, 35, 46, 0.9) !important;
        }}
        div[data-testid="stButton"] > button:hover {{
            border-color: var(--accent-blue) !important;
            color: #ffffff !important;
            background-color: var(--accent-blue-soft) !important;
        }}
        div[data-testid="stButton"] > button[kind="primary"] {{
            background-color: var(--accent-blue-dark) !important;
            border-color: var(--accent-blue-dark) !important;
            color: #ffffff !important;
        }}

        /* ========== DATAFRAMES ========== */
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background-color: rgba(35, 35, 46, 0.9) !important;
        }}
        div[data-testid="stDataFrame"] * {{
            color: #ffffff !important;
        }}

        /* ========== INPUTS HORS SIDEBAR ========== */
        input, select, textarea {{
            color: #ffffff !important;
            background-color: rgba(35, 35, 46, 0.95) !important;
        }}

        /* ========== SEPARATEURS ========== */
        hr {{ border-color: var(--border-color) !important; }}

        /* ========== EXPANDER ========== */
        details {{
            background-color: rgba(35, 35, 46, 0.9) !important;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        details summary {{
            color: #ffffff !important;
            font-weight: 600;
        }}

        /* ========== CAPTIONS ========== */
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: var(--text-muted) !important;
        }}

        /* ========== ONGLETS ========== */
        button[data-baseweb="tab"] {{
            color: #ffffff !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--accent-blue) !important;
        }}

        /* ========== MESSAGES STREAMLIT ========== */
        div[data-testid="stAlert"] {{
            background-color: rgba(35, 35, 46, 0.95) !important;
            color: #ffffff !important;
        }}
        div[data-testid="stAlert"] * {{
            color: #ffffff !important;
        }}

        /* ========== RADIO / CHECKBOX HORS SIDEBAR ========== */
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {{
            color: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def forcer_sidebar_ouverte():
    """Force la sidebar a rester ouverte, meme si l'utilisateur l'a fermee avant."""
    components.html(
        """
        <script>
        const openSidebar = () => {
            try {
                const doc = window.parent.document;
                const sb = doc.querySelector('section[data-testid="stSidebar"]');
                if (sb) {
                    sb.style.setProperty('transform', 'none', 'important');
                    sb.style.setProperty('margin-left', '0', 'important');
                    sb.style.setProperty('visibility', 'visible', 'important');
                    sb.style.setProperty('display', 'block', 'important');
                    sb.style.setProperty('min-width', '280px', 'important');
                }
            } catch (e) { console.log('sidebar:', e); }
        };
        // Reessayer plusieurs fois car Streamlit reconstruit le DOM
        setTimeout(openSidebar, 100);
        setTimeout(openSidebar, 500);
        setTimeout(openSidebar, 1200);
        setTimeout(openSidebar, 2500);
        </script>
        """,
        height=0,
    )


def section(titre, description=None):
    """Titre de section professionnel avec barre d'accent bleue."""
    st.markdown(
        f"""
        <div class="section-block">
            <div class="section-title">{titre}</div>
            <div class="section-bar"></div>
            {f'<div class="section-desc">{description}</div>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(html_content, warning=False):
    classe = "callout callout-warning" if warning else "callout"
    st.markdown(f'<div class="{classe}">{html_content}</div>', unsafe_allow_html=True)


def legend_item(color, label, texte):
    st.markdown(
        f"""
        <div class="legend-row">
            <div class="swatch" style="background-color:{color};"></div>
            <div class="legend-text"><b>{label}</b> &mdash; {texte}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------
@st.cache_data
def charger_donnees(chemin):
    return pd.read_csv(chemin, encoding="latin-1")


@st.cache_data
def charger_annotations(chemin):
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


def extraire_crises(annot, id_patient):
    id_col = annot.columns[0]
    ids = annot[id_col].astype(str).str.strip()
    ligne = annot[ids == id_patient]
    if ligne.empty:
        return None
    ligne = ligne.iloc[0]

    paires = []
    for i in range(1, 13):
        col_deb = next((c for c in annot.columns
                        if re.fullmatch(rf"(Deb|Debut|Debt)CE{i}", str(c))), None)
        col_fin = next((c for c in annot.columns
                        if re.fullmatch(rf"FinCE{i}", str(c))), None)
        if col_deb is None or col_fin is None:
            continue
        deb = pd.to_numeric(ligne[col_deb], errors="coerce")
        fin = pd.to_numeric(ligne[col_fin], errors="coerce")
        if pd.notna(deb) and pd.notna(fin):
            paires.append({"Crise": i, "Debut_brut": float(deb), "Fin_brut": float(fin)})
    return pd.DataFrame(paires, columns=["Crise", "Debut_brut", "Fin_brut"])


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


def marquer_artefacts(dataset, journal, marge):
    ons = np.sort(journal.loc[journal["Type"] == "artefact", "Onset"].values)
    if len(ons) == 0:
        return np.zeros(len(dataset), dtype=bool)
    debut_f = dataset["Window_start_s"].values - marge
    fin_f = dataset["Window_end_s"].values + marge
    return np.searchsorted(ons, fin_f, side="right") > np.searchsorted(ons, debut_f, side="left")


@st.cache_data(show_spinner="Detection du pic spectral...")
def detecter_pic(chemin, canal, fs):
    df = charger_donnees(chemin)
    signal = df[canal].values
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=int(fs), noverlap=int(fs * 0.5))
    mask = (f >= 2.5) & (f <= 4)
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


@st.cache_data(show_spinner="Construction du dataset de fenetres...")
def construire_dataset(chemin, canal, fs, window_sec, overlap, intervalles):
    df = charger_donnees(chemin)
    t = df["Time"].values
    x = df[canal].values.astype(float)

    y_sample = np.zeros(len(t))
    for deb, fin in intervalles:
        y_sample[(t >= deb) & (t <= fin)] = 1.0

    win = int(window_sec * fs)
    step = max(1, int(win * (1 - overlap)))
    if len(x) < win:
        return None

    W = sliding_window_view(x, win)[::step]
    Y = sliding_window_view(y_sample, win)[::step]
    starts = np.arange(len(W)) * step

    blocs = []
    for s in range(0, len(W), 4000):
        blocs.append(pd.DataFrame(features_fenetres(W[s:s + 4000], fs)))
    dataset = pd.concat(blocs, ignore_index=True)

    dataset.insert(0, "Window", np.arange(1, len(dataset) + 1))
    dataset["Class"] = (Y.mean(axis=1) >= 0.5).astype(int)
    dataset["Window_start_s"] = t[starts]
    dataset["Window_end_s"] = t[starts + win - 1]
    return dataset


# ------------------------------------------------------------
# Theme applique + forcage sidebar
# ------------------------------------------------------------
charger_theme()
forcer_sidebar_ouverte()

# ------------------------------------------------------------
# Barre d'etapes (TOUT EN HAUT)
# ------------------------------------------------------------
if "etape" not in st.session_state:
    st.session_state.etape = 0

cols_etapes = st.columns(len(STEPS))
for i, (col, nom_etape) in enumerate(zip(cols_etapes, STEPS)):
    with col:
        actif = " step-btn-active" if st.session_state.etape == i else ""
        st.markdown(f'<div class="step-btn{actif}">', unsafe_allow_html=True)
        if st.button(f"{i + 1}. {nom_etape}", key=f"step_{i}", use_container_width=True):
            st.session_state.etape = i
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Bandeau de titre
# ------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="eyebrow">Mini-projet Data Mining</div>
            <div class="app-title">Detection Automatique<br>des Crises d'Absence</div>
            <div class="app-subtitle">Analyse EEG &mdash; segmentation, features et classification</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Selection du patient / enregistrement
# ------------------------------------------------------------
if not os.path.exists(DOSSIER_DATA):
    st.error(f"Le dossier '{DOSSIER_DATA}' n'existe pas. Lancez d'abord le script de conversion.")
    st.stop()

fichiers_disponibles = sorted(f for f in os.listdir(DOSSIER_DATA) if f.endswith("d.csv"))
if not fichiers_disponibles:
    st.error("Aucun fichier CSV trouve.")
    st.stop()

st.sidebar.markdown('<div class="section-title" style="font-size:1rem;">Parametres</div>', unsafe_allow_html=True)
fichier_selectionne = st.sidebar.selectbox("Choisir un enregistrement", fichiers_disponibles)
patient_id = fichier_selectionne.replace("d.csv", "")
id_annot = patient_id.split("_0000")[0].replace("_", "-")

chemin_fichier = os.path.join(DOSSIER_DATA, fichier_selectionne)
df = charger_donnees(chemin_fichier)

fs = 1 / df["Time"].diff().median()
duree_totale = df["Time"].iloc[-1] - df["Time"].iloc[0]

section(f"Patient : {patient_id}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Lignes", f"{df.shape[0]:,}")
col2.metric("Colonnes", df.shape[1])
col3.metric("Frequence (Hz)", f"{fs:.2f}")
col4.metric("Duree (s)", f"{duree_totale:.1f}")

n_nan = int(df.isna().sum().sum())
if n_nan > 0:
    callout(f"<b>Valeurs manquantes detectees :</b> {n_nan} dans ce fichier.", warning=True)

canaux_eeg = [c for c in df.columns if c.startswith("EEG")]

# ------------------------------------------------------------
# Barre laterale
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-title" style="font-size:1rem;">Affichage</div>', unsafe_allow_html=True)
canaux_selectionnes = st.sidebar.multiselect("Canaux a afficher", canaux_eeg, default=canaux_eeg[:4])
duree_max = int(df["Time"].iloc[-1])
debut = st.sidebar.slider("Debut (secondes)", 0, max(duree_max - 1, 1), 0)
duree = st.sidebar.slider("Duree (secondes)", 1, 30, 5)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-title" style="font-size:1rem;">Analyse</div>', unsafe_allow_html=True)
canal_analyse = st.sidebar.selectbox("Canal d'analyse", canaux_eeg)
window_sec = st.sidebar.slider("Taille de fenetre (s)", 1, 10, 2)
overlap = round(st.sidebar.slider("Recouvrement", 0.0, 0.9, 0.5, 0.1), 2)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-title" style="font-size:1rem;">Annotations</div>', unsafe_allow_html=True)
chemin_annot = st.sidebar.text_input("Fichier d'annotations",
                                     value=trouver_annotations(DOSSIER_DATA) or "")

annot, crises_brutes, crises, intervalles = None, None, None, ()
crises_toutes = None
t_spec = puiss = seuil = None
t_pic = None
erreur_annot = None

if chemin_annot and os.path.exists(chemin_annot):
    try:
        annot = charger_annotations(chemin_annot)
        crises_brutes = extraire_crises(annot, id_annot)
    except Exception as e:
        erreur_annot = str(e)

if crises_brutes is not None and not crises_brutes.empty:
    t_spec, puiss, t_pic_auto, seuil = detecter_pic(chemin_fichier, canal_analyse, fs)
    t_pic = st.sidebar.number_input(
        "Instant du pic de crise dans l'EEG (s)",
        min_value=0.0, max_value=float(duree_totale),
        value=round(t_pic_auto, 1), step=0.5,
        key=f"tpic_{fichier_selectionne}_{canal_analyse}",
        help="Detecte automatiquement (pic 2.5-4 Hz) et suppose coincider avec le debut "
             "de la crise choisie ci-dessous. Ajustez-le si la zone rouge ne colle pas au signal.",
    )
    crise_ref = st.sidebar.selectbox(
        "Crise de l'Excel correspondant au pic", crises_brutes["Crise"].tolist(),
        key=f"cref_{fichier_selectionne}",
        help="Par defaut la 1ere. Si ce fichier ne contient que les crises suivantes "
             "(ex. fichier _0001), choisissez la crise qui correspond au pic.")
    ligne_ref = crises_brutes[crises_brutes["Crise"] == crise_ref].iloc[0]

    offset_s = ligne_ref["Debut_brut"] * SECONDES_PAR_JOUR - t_pic
    crises_toutes = crises_brutes.copy()
    crises_toutes["Debut_s"] = crises_toutes["Debut_brut"] * SECONDES_PAR_JOUR - offset_s
    crises_toutes["Fin_s"] = crises_toutes["Fin_brut"] * SECONDES_PAR_JOUR - offset_s
    crises_toutes["Duree_s"] = crises_toutes["Fin_s"] - crises_toutes["Debut_s"]

    t_min, t_max = float(df["Time"].iloc[0]), float(df["Time"].iloc[-1])
    dans_fichier = (crises_toutes["Fin_s"] >= t_min) & (crises_toutes["Debut_s"] <= t_max)
    crises = crises_toutes[dans_fichier].reset_index(drop=True)
    intervalles = tuple((round(float(a), 3), round(float(b), 3))
                        for a, b in zip(crises["Debut_s"], crises["Fin_s"]))

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-title" style="font-size:1rem;">Journal (fichier h)</div>', unsafe_allow_html=True)
chemin_journal = os.path.join(DOSSIER_DATA, fichier_selectionne[:-len("d.csv")] + "h.csv")
journal = charger_journal(chemin_journal)
exclure_art, marge_art = False, 1.0
if journal is None:
    st.sidebar.caption(f"Pas de journal trouve ({os.path.basename(chemin_journal)}).")
else:
    st.sidebar.caption(f"{len(journal)} notes chargees.")
    exclure_art = st.sidebar.checkbox(
        "Exclure les fenetres avec artefact", value=False,
        help="Retire du dataset les fenetres proches d'une note ARTEFACT / BOUGE / SATURATION.")
    marge_art = st.sidebar.slider("Marge autour d'un artefact (s)", 0.0, 5.0, 1.0, 0.5)


def get_dataset_brut():
    if crises is None:
        return None
    ds = construire_dataset(chemin_fichier, canal_analyse, fs, window_sec, overlap, intervalles)
    if ds is None:
        return None
    ds = ds.copy()
    if journal is not None:
        ds["Artefact"] = marquer_artefacts(ds, journal, marge_art)
    return ds


def get_dataset():
    ds = get_dataset_brut()
    if ds is not None and exclure_art and "Artefact" in ds.columns:
        ds = ds[~ds["Artefact"]].reset_index(drop=True)
    return ds


def message_annotations_indisponibles():
    if erreur_annot:
        st.error(f"Impossible de lire le fichier d'annotations : {erreur_annot}")
    elif not chemin_annot or not os.path.exists(chemin_annot):
        callout("Indiquez le chemin de <b>Annotations.xlsx</b> dans la barre laterale.")
    elif crises_brutes is None:
        proches = [str(i) for i in annot.iloc[:, 0].dropna().astype(str)
                   if i.startswith(id_annot[:7])]
        callout(f"ID <b>{id_annot}</b> introuvable dans les annotations. IDs proches : {proches}", warning=True)
    else:
        callout(f"Aucune crise annotee pour <b>{id_annot}</b>.", warning=True)


# ------------------------------------------------------------
# Contenu selon l'etape active
# ------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
etape_active = STEPS[st.session_state.etape]

# ---------- Etape 1 : signal ----------
if etape_active == "Signal":
    section(f"Signal EEG", f"{debut}s a {debut + duree}s")
    if canaux_selectionnes:
        debut_idx = int(debut * fs)
        fin_idx = int((debut + duree) * fs)

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#1a1a24")
        ax.set_facecolor("#1a1a24")
        for i, canal in enumerate(canaux_selectionnes):
            signal = df[canal].iloc[debut_idx:fin_idx].values
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            ax.plot(df["Time"].iloc[debut_idx:fin_idx], signal_norm + i * 3,
                    label=canal, linewidth=0.8)

        if crises is not None:
            premier = True
            for _, c in crises.iterrows():
                if c["Fin_s"] >= debut and c["Debut_s"] <= debut + duree:
                    ax.axvspan(c["Debut_s"], c["Fin_s"], color="crimson", alpha=0.25,
                               label="Crise (annotation)" if premier else None)
                    premier = False

        if journal is not None:
            styles = {"artefact": ("gray", "-", "Artefact (journal)"),
                      "crise": ("darkorange", "--", "Note crise/absence (journal)")}
            deja = set()
            visibles = journal[(journal["Onset"] >= debut) & (journal["Onset"] <= debut + duree)]
            for _, ev in visibles.iterrows():
                if ev["Type"] not in styles:
                    continue
                couleur, trait, nom = styles[ev["Type"]]
                ax.axvline(ev["Onset"], color=couleur, ls=trait, lw=1.6, alpha=0.8,
                           label=nom if nom not in deja else None)
                deja.add(nom)

        ax.set_xlabel("Temps (secondes)", color="#ffffff")
        ax.set_ylabel("Amplitude (normalisee + decalage)", color="#ffffff")
        ax.set_yticks([])
        ax.tick_params(colors="#ffffff")
        ax.grid(True, linestyle="--", alpha=0.3, color="#5a5a6a")
        ax.legend(loc="upper right", fontsize=8, ncol=2, facecolor="#23232e", edgecolor="#3a3a48", labelcolor="#ffffff")
        st.pyplot(fig)
        plt.close(fig)

        section("Legende du graphique")
        legend_item("#5b9cff", "Courbes", "Chaque couleur represente un canal EEG (electrode). Decalees verticalement.")
        legend_item("#dc2626", "Zone rouge translucide (crise)", "Periode de crise issue du fichier Annotations.xlsx.")
        legend_item("#9ca3af", "Ligne grise verticale (artefact)", "Evenement indesirable note par le technicien.")
        legend_item("#f59e0b", "Ligne orange pointillee (note crise/absence)", "Suspicion manuelle du technicien.")
        callout(
            "<b>Comment lire le graphique :</b>"
            "<ul>"
            "<li>Axe horizontal (X) : le temps en secondes.</li>"
            "<li>Axe vertical (Y) : amplitude normalisee (chaque canal decale de 3 unites).</li>"
            "<li>Une crise d'absence typique se manifeste par des oscillations rapides et regulieres "
            "(pointes-ondes a environ 3 Hz) simultanees sur plusieurs canaux.</li>"
            "</ul>"
        )

        if journal is not None:
            ev = journal[(journal["Onset"] >= debut) & (journal["Onset"] <= debut + duree)]
            if not ev.empty:
                st.caption("Notes du journal dans cette fenetre :")
                st.dataframe(ev.assign(Annotations=ev["Annotations"].str[:80])
                             [["Onset", "Type", "Annotations"]], use_container_width=True)
    else:
        callout("Selectionnez au moins un canal.", warning=True)

    section("Statistiques descriptives")
    if canaux_selectionnes:
        st.dataframe(df[canaux_selectionnes].describe().T, use_container_width=True)
        callout(
            "<b>Signification des statistiques :</b>"
            "<ul>"
            "<li><b>count</b> : nombre de valeurs (echantillons) analysees.</li>"
            "<li><b>mean</b> : valeur moyenne du signal sur la periode.</li>"
            "<li><b>std</b> : ecart-type (dispersion des valeurs autour de la moyenne).</li>"
            "<li><b>min / max</b> : valeurs minimale et maximale.</li>"
            "<li><b>25% / 50% / 75%</b> : quartiles (25 %, mediane, 75 %).</li>"
            "</ul>"
        )

    if journal is not None:
        section("Journal du technicien (fichier h)")
        couleurs = {"artefact": "gray", "crise": "crimson", "yeux": "steelblue",
                    "hpn": "green", "sli": "purple", "autre": "black"}
        fig, ax = plt.subplots(figsize=(14, 2.8))
        fig.patch.set_facecolor("#1a1a24")
        ax.set_facecolor("#1a1a24")
        for k, (typ, coul) in enumerate(couleurs.items()):
            o = journal.loc[journal["Type"] == typ, "Onset"]
            ax.scatter(o, [k] * len(o), color=coul, s=25)
        if crises is not None:
            for _, c in crises.iterrows():
                ax.axvspan(c["Debut_s"], c["Fin_s"], color="crimson", alpha=0.2)
        ax.set_yticks(range(len(couleurs)))
        ax.set_yticklabels(list(couleurs), color="#ffffff")
        ax.set_xlabel("Temps (s)", color="#ffffff")
        ax.tick_params(colors="#ffffff")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3, color="#5a5a6a")
        st.pyplot(fig)
        plt.close(fig)

        section("Legende du journal")
        legend_item("#9ca3af", "artefact", "Mouvement du patient, saturation du signal, bruit electrique. A exclure de l'analyse.")
        legend_item("#dc2626", "crise", "Suspicion de crise d'absence notee par le technicien (observation visuelle).")
        legend_item("#4682b4", "yeux", "Ouverture/fermeture des yeux (YO / YF). Cree des artefacts frontaux typiques.")
        legend_item("#16a34a", "hpn", "Hyperventilation (epreuve de provocation utilisee pour declencher des crises).")
        legend_item("#7c3aed", "sli", "Stimulation Lumineuse Intermittente (flashs lumineux pour provoquer des crises).")
        legend_item("#111827", "autre", "Note diverse (debut d'enregistrement, changement de parametre, etc.).")
        callout("Les zones rouges translucides en arriere-plan representent les crises issues du fichier Excel Annotations.xlsx (reference).")

        with st.expander("Voir toutes les notes"):
            st.dataframe(journal.assign(Annotations=journal["Annotations"].str[:100])
                         [["Onset", "Type", "Annotations"]], use_container_width=True)

# ---------- Etape 2 : annotations ----------
elif etape_active == "Crises & cible":
    if crises is None:
        message_annotations_indisponibles()
    else:
        section("Crises annotees", "Valeurs Excel brutes = fraction de jour")
        st.dataframe(crises_brutes, use_container_width=True)
        callout(
            "<b>Signification des colonnes :</b>"
            "<ul>"
            "<li><b>Crise</b> : numero de la crise (1, 2, 3...).</li>"
            "<li><b>Debut_brut</b> : heure de debut au format Excel (fraction de jour, ex. 0.5 = 12h00).</li>"
            "<li><b>Fin_brut</b> : heure de fin au format Excel.</li>"
            "</ul>"
        )

        section("Detection spectrale", f"Puissance 2.5-4 Hz ({canal_analyse})")
        fig, ax = plt.subplots(figsize=(14, 3.5))
        fig.patch.set_facecolor("#1a1a24")
        ax.set_facecolor("#1a1a24")
        ax.plot(t_spec, puiss, color="crimson", lw=0.8)
        ax.axhline(seuil, color="k", ls="--", label="seuil mu+3sigma")
        ax.axvline(t_pic, color="green", ls=":", label=f"t_pic = {t_pic:.1f} s")
        ax.set_xlabel("Temps (s)", color="#ffffff")
        ax.set_ylabel("Puissance 2.5-4 Hz", color="#ffffff")
        ax.tick_params(colors="#ffffff")
        ax.legend(facecolor="#23232e", edgecolor="#3a3a48", labelcolor="#ffffff")
        st.pyplot(fig)
        plt.close(fig)

        callout(
            "<b>Comprendre le spectre de puissance :</b>"
            "<ul>"
            "<li>Courbe rouge : puissance du signal dans la bande 2.5-4 Hz, typique des crises d'absence "
            "(pointes-ondes a environ 3 Hz).</li>"
            "<li>Ligne noire pointillee (seuil mu+3sigma) : seuil statistique = moyenne + 3 ecarts-types.</li>"
            "<li>Ligne verte pointillee (t_pic) : instant du pic maximum de puissance 2.5-4 Hz, point d'ancrage "
            "pour synchroniser les annotations.</li>"
            f"<li><b>Ratio pic/seuil actuel :</b> {puiss.max() / seuil:.2f} &mdash; un ratio superieur a 1 indique "
            "une activite 2.5-4 Hz nettement au-dessus du bruit de fond.</li>"
            "</ul>"
        )

        section("Crises converties en temps relatif au debut de l'EEG")
        affichage = crises_toutes[["Crise", "Debut_s", "Fin_s", "Duree_s"]].round(2)
        affichage["Dans ce fichier"] = np.where(crises_toutes["Crise"].isin(crises["Crise"]),
                                                "Oui", "Non")
        st.dataframe(affichage, use_container_width=True)
        callout(
            "<b>Signification des colonnes :</b>"
            "<ul>"
            "<li><b>Debut_s / Fin_s</b> : heure convertie en secondes depuis le debut de l'enregistrement EEG.</li>"
            "<li><b>Duree_s</b> : duree de la crise en secondes.</li>"
            "<li><b>Dans ce fichier</b> : indique si la crise tombe dans l'enregistrement courant, "
            "ou si elle appartient a un autre fichier du meme patient.</li>"
            "</ul>"
        )

        if crises.empty:
            callout("Aucune crise de l'Excel ne tombe dans cet enregistrement. Verifiez t_pic et la crise "
                    "choisie dans la barre laterale.", warning=True)
        else:
            callout(f"<b>{len(crises)} crise(s) sur {len(crises_toutes)}</b> sont dans cet enregistrement.")

        section("Verification avec le journal du technicien")
        if journal is None:
            st.caption("Aucun fichier h trouve pour cet enregistrement.")
        else:
            art = np.sort(journal.loc[journal["Type"] == "artefact", "Onset"].values)
            notes = np.sort(journal.loc[journal["Type"] == "crise", "Onset"].values)
            verif = crises[["Crise", "Debut_s", "Fin_s"]].copy()
            verif["Notes crise/absence (+/-30 s)"] = [
                int(((notes >= d - 30) & (notes <= f + 30)).sum())
                for d, f in zip(crises["Debut_s"], crises["Fin_s"])]
            verif["Artefacts pendant la crise"] = [
                int(((art >= d) & (art <= f)).sum())
                for d, f in zip(crises["Debut_s"], crises["Fin_s"])]
            st.dataframe(verif.round(2), use_container_width=True)

            if len(art):
                d_art = float(np.min(np.abs(art - t_pic)))
                if d_art <= 3:
                    callout(f"Le pic (t = {t_pic:.1f} s) est a {d_art:.1f} s d'une note ARTEFACT/BOUGE : "
                            "la detection est peut-etre tombee sur un mouvement et non sur la crise. "
                            "Verifiez l'alignement.", warning=True)
                else:
                    callout(f"Le pic est a {d_art:.0f} s de l'artefact le plus proche : pas de confusion apparente.")
            if len(notes):
                st.caption("Notes crise/absence du technicien : " + ", ".join(f"{n:.0f} s" for n in notes))
            else:
                st.caption("Le technicien n'a note aucune crise/absence dans ce journal.")

        section("Signal complet avec crises positionnees")
        pas = max(1, len(df) // 200_000)
        sub = df.iloc[::pas]
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("#1a1a24")
        ax.set_facecolor("#1a1a24")
        ax.plot(sub["Time"], sub[canal_analyse], lw=0.5, color="steelblue")
        for k, (_, c) in enumerate(crises.iterrows()):
            ax.axvspan(c["Debut_s"], c["Fin_s"], color="crimson", alpha=0.4,
                       label="Crise (annotation)" if k == 0 else None)
        ax.set_xlabel("Temps (s)", color="#ffffff")
        ax.set_ylabel("Amplitude EEG", color="#ffffff")
        ax.tick_params(colors="#ffffff")
        if not crises.empty:
            ax.legend(facecolor="#23232e", edgecolor="#3a3a48", labelcolor="#ffffff")
        st.pyplot(fig)
        plt.close(fig)
        callout(
            "Courbe bleue : signal EEG complet (sous-echantillonne pour l'affichage). "
            "Zones rouges translucides : periodes de crise annotees, positionnees sur le signal. "
            "<b>Objectif :</b> verifier visuellement que les zones rouges correspondent bien a des bouffees "
            "d'activite anormale sur le signal."
        )

        # Repartition de la variable cible
        y_sample = np.zeros(len(df), dtype=int)
        for deb, fin in intervalles:
            y_sample[((df["Time"] >= deb) & (df["Time"] <= fin)).values] = 1
        rep = pd.Series(y_sample).value_counts().reindex([0, 1], fill_value=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Echantillons hors crise (0)", f"{rep[0]:,}")
        c2.metric("Echantillons en crise (1)", f"{rep[1]:,}")
        c3.metric("Proportion de crise", f"{rep[1] / len(df) * 100:.3f} %")
        callout(
            "<b>Variable cible (Class) :</b> 0 = echantillon hors crise (etat normal) ; "
            "1 = echantillon pendant une crise. La proportion de crise est tres faible (moins de 1 %) : "
            "c'est un probleme de classification desequilibree."
        )

# ---------- Etape 3 : features ----------
elif etape_active == "Features":
    dataset = get_dataset()
    if dataset is None:
        message_annotations_indisponibles()
    else:
        section("Vue d'ensemble du dataset")
        n1 = int((dataset["Class"] == 1).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Fenetres totales", f"{len(dataset):,}")
        c2.metric("Classe 0 (pas de crise)", f"{len(dataset) - n1:,}")
        c3.metric("Classe 1 (crise)", f"{n1:,}")
        st.caption(f"Fenetres de {window_sec} s, recouvrement {overlap * 100:.0f} %, "
                   f"canal {canal_analyse}. Une fenetre est 'crise' si au moins 50 % de ses points le sont.")

        brut = get_dataset_brut()
        if journal is not None and brut is not None:
            n_art = int(brut["Artefact"].sum())
            n_art_crise = int(brut.loc[brut["Class"] == 1, "Artefact"].sum())
            st.caption(f"{n_art} fenetres touchent un artefact du journal (dont {n_art_crise} en crise). "
                       + ("Elles sont exclues du dataset." if exclure_art else
                          "Cochez 'Exclure les fenetres avec artefact' pour les retirer."))

        if n1 == 0:
            callout("Aucune fenetre etiquetee 'crise' : les annotations sont probablement mal synchronisees "
                    "(voir t_pic dans la barre laterale).", warning=True)
        else:
            st.dataframe(dataset.head(20), use_container_width=True)

            section("Moyenne des features par classe")
            comp = dataset.groupby("Class")[FEATURE_COLS].mean().T
            comp.columns = [f"Classe {c} ({'crise' if c == 1 else 'pas crise'})" for c in comp.columns]
            st.dataframe(comp, use_container_width=True)

            section("Signification des features")
            callout(
                "<b>Features temporelles</b>"
                "<ul>"
                "<li><b>Mean</b> : moyenne du signal dans la fenetre.</li>"
                "<li><b>Std</b> : ecart-type (dispersion).</li>"
                "<li><b>Variance</b> : carre de l'ecart-type.</li>"
                "<li><b>Min / Max</b> : valeurs minimale et maximale.</li>"
                "<li><b>Amplitude</b> : Max moins Min (etendue du signal).</li>"
                "<li><b>RMS</b> : Root Mean Square (energie du signal).</li>"
                "<li><b>Energy</b> : somme des carres des valeurs.</li>"
                "</ul>"
                "<b>Features frequentielles (puissance par bande)</b>"
                "<ul>"
                "<li><b>Power_Delta</b> (0.5-4 Hz) : sommeil profond, pathologie.</li>"
                "<li><b>Power_Theta</b> (4-8 Hz) : somnolence, enfance.</li>"
                "<li><b>Power_Alpha</b> (8-13 Hz) : eveil calme, yeux fermes.</li>"
                "<li><b>Power_Beta</b> (13-30 Hz) : concentration, activite mentale.</li>"
                "</ul>"
                "Les crises d'absence augmentent generalement la puissance dans la bande Delta/Theta (2.5-4 Hz)."
            )

            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("#1a1a24")
            ax.set_facecolor("#1a1a24")
            comp.plot(kind="bar", ax=ax)
            ax.set_yscale("symlog")
            ax.set_ylabel("Valeur moyenne (echelle symlog)", color="#ffffff")
            ax.tick_params(colors="#ffffff")
            plt.xticks(rotation=45, color="#ffffff")
            plt.yticks(color="#ffffff")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Comparaison des features moyennes entre les deux classes. L'echelle symlog permet de "
                       "comparer des valeurs d'ordres de grandeur differents.")

            section("Matrice de correlation entre features")
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor("#1a1a24")
            ax.set_facecolor("#1a1a24")
            sns.heatmap(dataset[FEATURE_COLS].corr(), annot=True, fmt=".2f",
                        cmap="coolwarm", center=0, ax=ax, annot_kws={"color": "white"})
            ax.tick_params(colors="#ffffff")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            callout(
                "<b>Comprendre la heatmap :</b>"
                "<ul>"
                "<li>Rouge fonce (+1) : correlation positive forte.</li>"
                "<li>Bleu fonce (-1) : correlation negative forte.</li>"
                "<li>Blanc (0) : aucune correlation lineaire.</li>"
                "</ul>"
                "Si deux features sont tres correlees (plus de 0.9), elles apportent la meme information : "
                "on peut en supprimer une pour simplifier le modele et eviter la multicolinearite."
            )

        st.download_button("Telecharger le dataset (CSV)",
                           dataset.to_csv(index=False).encode("utf-8"),
                           file_name=f"dataset_{patient_id}.csv", mime="text/csv")

# ---------- Etape 4 : modelisation ----------
elif etape_active == "Modeles":
    dataset = get_dataset()
    if dataset is None:
        message_annotations_indisponibles()
    elif dataset["Class"].sum() < 2:
        callout("Pas assez de fenetres 'crise' pour entrainer un modele.", warning=True)
    else:
        section("Entrainement des modeles")
        callout(
            "Avec un seul enregistrement, les scores sont a prendre avec prudence. Avec un recouvrement de "
            "50 %, le decoupage aleatoire met des fenetres voisines dans train ET test (fuite de donnees). "
            "Le decoupage chronologique est plus honnete.",
            warning=True,
        )

        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Part de test", 0.1, 0.5, 0.2, 0.05)
        mode = c2.radio("Decoupage", ["Aleatoire stratifie (comme le notebook)",
                                      "Chronologique (sans fuite)"])
        balanced = c3.checkbox("Ponderer les classes (class_weight='balanced')", value=False,
                               help="Utile car les crises sont rares. Non applicable au KNN.")

        if st.button("Entrainer les modeles"):
            X, y = dataset[FEATURE_COLS], dataset["Class"]
            try:
                if mode.startswith("Aleatoire"):
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y)
                else:
                    cut = int(len(X) * (1 - test_size))
                    X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
            except ValueError as e:
                st.error(f"Decoupage impossible : {e}")
                st.stop()

            if y_tr.nunique() < 2:
                st.error("Le jeu d'entrainement ne contient qu'une seule classe avec ce decoupage.")
                st.stop()

            scaler = StandardScaler()
            X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

            cw = "balanced" if balanced else None
            modeles = {
                "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight=cw), False),
                "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42,
                                                         class_weight=cw, n_jobs=-1), False),
                "KNN": (KNeighborsClassifier(n_neighbors=5), True),
                "SVM": (SVC(kernel="rbf", random_state=42, class_weight=cw), True),
            }

            lignes, preds, importances = [], {}, None
            with st.spinner("Entrainement en cours..."):
                for nom, (modele, scale) in modeles.items():
                    Xa, Xb = (X_tr_s, X_te_s) if scale else (X_tr, X_te)
                    modele.fit(Xa, y_tr)
                    y_pred = modele.predict(Xb)
                    preds[nom] = y_pred
                    lignes.append({
                        "Modele": nom,
                        "Accuracy": accuracy_score(y_te, y_pred),
                        "Precision": precision_score(y_te, y_pred, zero_division=0),
                        "Recall": recall_score(y_te, y_pred, zero_division=0),
                        "F1-score": f1_score(y_te, y_pred, zero_division=0),
                    })
                    if nom == "Random Forest":
                        importances = pd.Series(modele.feature_importances_, index=FEATURE_COLS)

            st.session_state["resultats"] = {
                "df": pd.DataFrame(lignes).set_index("Modele").round(3),
                "preds": preds, "y_te": y_te, "importances": importances,
                "n_train": len(y_tr), "n_test": len(y_te),
                "crises_train": int(y_tr.sum()), "crises_test": int(y_te.sum()),
            }

        res = st.session_state.get("resultats")
        if res:
            callout(f"<b>Train :</b> {res['n_train']} fenetres ({res['crises_train']} crise) &mdash; "
                    f"<b>Test :</b> {res['n_test']} fenetres ({res['crises_test']} crise)")
            st.dataframe(res["df"], use_container_width=True)

            section("Signification des metriques")
            callout(
                "<ul>"
                "<li><b>Accuracy</b> : proportion de predictions correctes (peut etre trompeuse si les "
                "classes sont desequilibrees).</li>"
                "<li><b>Precision</b> : parmi les fenetres predites 'crise', combien le sont vraiment "
                "(evite les fausses alertes).</li>"
                "<li><b>Recall</b> : parmi les vraies crises, combien ont ete detectees (evite de rater "
                "des crises).</li>"
                "<li><b>F1-score</b> : moyenne harmonique de Precision et Recall.</li>"
                "</ul>"
                "Sur des donnees desequilibrees, privilegiez le Recall et le F1-score plutot que l'Accuracy."
            )

            section("Matrices de confusion")
            cols = st.columns(len(res["preds"]))
            for col, (nom, y_pred) in zip(cols, res["preds"].items()):
                fig, ax = plt.subplots(figsize=(3, 3))
                fig.patch.set_facecolor("#1a1a24")
                ax.set_facecolor("#1a1a24")
                ConfusionMatrixDisplay.from_predictions(
                    res["y_te"], y_pred, ax=ax, colorbar=False,
                    display_labels=["Normal", "Crise"])
                ax.set_title(nom, fontsize=10, color="#ffffff")
                ax.tick_params(colors="#ffffff")
                plt.tight_layout()
                col.pyplot(fig)
                plt.close(fig)

            callout(
                "<b>Comprendre la matrice de confusion :</b>"
                "<ul>"
                "<li><b>Vrai negatif</b> : bien classe comme normal.</li>"
                "<li><b>Vrai positif</b> : crise correctement detectee.</li>"
                "<li><b>Faux positif</b> : fausse alerte (predit crise, mais c'etait normal).</li>"
                "<li><b>Faux negatif</b> : crise manquee (predit normal, mais c'etait une crise) &mdash; "
                "le cas le plus critique.</li>"
                "</ul>"
            )

            if res["importances"] is not None:
                section("Importance des features (Random Forest)")
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor("#1a1a24")
                ax.set_facecolor("#1a1a24")
                res["importances"].sort_values().plot.barh(ax=ax, color="#5b9cff")
                ax.tick_params(colors="#ffffff")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Les features en haut sont les plus utilisees par le modele pour distinguer les crises.")

# ------------------------------------------------------------
# Contexte physiologique (dossier image/)
# ------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("Contexte physiologique : le systeme nerveux"):
    section("Pourquoi ce rappel", "Le signal EEG mesure l'activite electrique produite par le systeme nerveux central.")
    c1, c2 = st.columns(2)
    with c1:
        chemin = os.path.join(IMAGE_DIR, "fonctions.png")
        if os.path.exists(chemin):
            st.image(chemin, use_container_width=True)
        chemin = os.path.join(IMAGE_DIR, "processus.png")
        if os.path.exists(chemin):
            st.image(chemin, use_container_width=True)
    with c2:
        chemin = os.path.join(IMAGE_DIR, "composants.png")
        if os.path.exists(chemin):
            st.image(chemin, use_container_width=True)
        chemin = os.path.join(IMAGE_DIR, "faits.png")
        if os.path.exists(chemin):
            st.image(chemin, use_container_width=True)
    callout(
        "Le cerveau, la moelle epiniere et les nerfs forment un reseau qui recoit, traite et transmet "
        "l'information. Les crises d'absence correspondent a une decharge synchrone et anormale de ce "
        "reseau, visible sur l'EEG sous forme de pointes-ondes a environ 3 Hz."
    )

st.caption("Mini-projet Data Mining - Detection des crises d'absence EEG")