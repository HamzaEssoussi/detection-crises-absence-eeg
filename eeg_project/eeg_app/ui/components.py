"""Petits composants HTML reutilisables (titres, encadres, legendes, messages)."""
import os

import streamlit as st


def section(titre, description=None):
    """Titre de section avec barre d'accent bleue."""
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


def message_annotations_indisponibles(ctx):
    """Explique pourquoi les crises annotees ne sont pas disponibles."""
    if ctx.erreur_annot:
        st.error(f"Impossible de lire le fichier d'annotations : {ctx.erreur_annot}")
    elif not ctx.chemin_annot or not os.path.exists(ctx.chemin_annot):
        callout("Indiquez le chemin de <b>Annotations.xlsx</b> dans la barre laterale.")
    elif ctx.crises_brutes is None:
        proches = [str(i) for i in ctx.annot.iloc[:, 0].dropna().astype(str)
                   if i.startswith(ctx.id_annot[:7])]
        callout(f"ID <b>{ctx.id_annot}</b> introuvable dans les annotations. IDs proches : {proches}",
                warning=True)
    else:
        callout(f"Aucune crise annotee pour <b>{ctx.id_annot}</b>.", warning=True)
