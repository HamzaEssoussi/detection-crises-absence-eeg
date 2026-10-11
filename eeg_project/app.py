import streamlit as st

from eeg_app.config import PAGE_TITLE
from eeg_app.steps import ETAPES
from eeg_app.ui import layout
from eeg_app.ui.context_builder import construire_contexte, construire_contexte_multi
from eeg_app.ui.theme import charger_theme

st.set_page_config(page_title=PAGE_TITLE, layout="wide",
                   initial_sidebar_state="expanded")


def main():
    charger_theme()

    etape_active = layout.barre_etapes()
    layout.entete()

    # Choix du mode
    mode_multi = st.sidebar.checkbox(
        "Mode multi-patients", value=False,
        help="Cochez pour traiter plusieurs patients simultanement.")

    if mode_multi:
        ctx = construire_contexte_multi()
    else:
        ctx = construire_contexte()

    layout.resume_patient(ctx)
    layout.separateur()
    ETAPES[etape_active](ctx)
    layout.separateur()
    layout.contexte_physiologique()
    layout.pied_de_page()


main()