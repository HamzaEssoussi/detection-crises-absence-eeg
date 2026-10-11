"""Elements de mise en page : barre d'etapes, bandeau, resume patient, pied de page."""
import os

import streamlit as st

from eeg_app.config import IMAGE_DIR, STEPS
from eeg_app.ui.components import callout, section


def barre_etapes():
    """Barre de navigation en haut. Retourne le nom de l'etape active."""
    if "etape" not in st.session_state:
        st.session_state.etape = 0

    cols = st.columns(len(STEPS))
    for i, (col, nom_etape) in enumerate(zip(cols, STEPS)):
        with col:
            actif = " step-btn-active" if st.session_state.etape == i else ""
            st.markdown(f'<div class="step-btn{actif}">', unsafe_allow_html=True)
            if st.button(f"{i + 1}. {nom_etape}", key=f"step_{i}", use_container_width=True,
                         type="primary" if actif else "secondary"):
                st.session_state.etape = i
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    return STEPS[st.session_state.etape]


def entete():
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


def resume_patient(ctx):
    section(f"Patient : {ctx.patient_id}")
    c1, c2, c3, c4 = st.columns(4)
    if getattr(ctx, "mode", "mono") == "multi":
        c1.metric("Patients", ctx.df["Patient_ID"].nunique())
        c2.metric("Fenetres", len(ctx.df))
        c3.metric("Canal analyse", ctx.canal_analyse)
        c4.metric("Taille fenetre (ms)", f"{ctx.window_sec * 1000:.0f}")
        return

    c1.metric("Lignes", str(ctx.df.shape[0]))
    c2.metric("Colonnes", ctx.df.shape[1])
    c3.metric("Frequence (Hz)", f"{ctx.fs:.2f}")
    c4.metric("Duree (s)", f"{ctx.duree_totale:.1f}")

    n_nan = int(ctx.df.isna().sum().sum())
    if n_nan > 0:
        callout(f"<b>Valeurs manquantes detectees :</b> {n_nan} dans ce fichier.", warning=True)


def separateur():
    st.markdown("<hr>", unsafe_allow_html=True)


def _image(nom):
    chemin = os.path.join(IMAGE_DIR, nom)
    if os.path.exists(chemin):
        st.image(chemin, use_container_width=True)


def contexte_physiologique():
    with st.expander("Contexte physiologique : le systeme nerveux"):
        section("Pourquoi ce rappel",
                "Le signal EEG mesure l'activite electrique produite par le systeme nerveux central.")
        c1, c2 = st.columns(2)
        with c1:
            _image("fonctions.png")
            _image("processus.png")
        with c2:
            _image("composants.png")
            _image("faits.png")
        callout(
            "Le cerveau, la moelle epiniere et les nerfs forment un reseau qui recoit, traite et transmet "
            "l'information. Les crises d'absence correspondent a une decharge synchrone et anormale de ce "
            "reseau, visible sur l'EEG sous forme de pointes-ondes a environ 3 Hz."
        )


def pied_de_page():
    st.caption("Mini-projet Data Mining - Detection des crises d'absence EEG")