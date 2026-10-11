"""Theme visuel de l'application (CSS externe)."""
import base64
import os

import streamlit as st

from eeg_app.config import ASSETS_DIR, IMAGE_DIR


def image_base64(nom_fichier):
    """Encode une image du dossier image/ en base64 (None si absente)."""
    chemin = os.path.join(IMAGE_DIR, nom_fichier)
    if not os.path.exists(chemin):
        return None
    with open(chemin, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _css_fond():
    """Construit les fonds de l'application et du bandeau depuis l'image hero."""
    bg_b64 = image_base64("banner_hero.png")
    if not bg_b64:
        return ""
    return f"""
    .stApp,
    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.38), rgba(255, 255, 255, 0.38)),
            url('data:image/png;base64,{bg_b64}') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }}
    [data-testid="stMain"] {{
        background-color: transparent !important;
    }}
    .app-header {{
        background-image:
            linear-gradient(rgba(33, 37, 41, 0.86), rgba(33, 37, 41, 0.92)),
            url('data:image/png;base64,{bg_b64}');
        background-size: cover;
        background-position: center;
    }}
    """


def charger_theme():
    """Injecte le CSS (assets/theme.css) + l'image de fond."""
    with open(os.path.join(ASSETS_DIR, "theme.css"), encoding="utf-8") as f:
        css = f.read().replace("/*__BACKGROUND__*/", _css_fond())
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)