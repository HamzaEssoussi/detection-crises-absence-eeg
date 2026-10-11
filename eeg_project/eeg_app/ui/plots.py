"""Toutes les figures Plotly interactives (theme de l application).

Zoom (molette / glisser), deplacement, survol avec valeurs, legende cliquable,
export PNG via la barre d'outils. Chaque fonction retourne une figure.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from eeg_app.config import (COULEUR_ACCENT, COULEUR_BORDURE, COULEUR_FOND,
                            COULEUR_GRILLE, COULEUR_PANEL, COULEUR_TEXTE)

CONFIG = {
    "scrollZoom": True,          # zoom a la molette
    "displaylogo": False,
    "displayModeBar": True,
    "doubleClick": "reset",      # double-clic = reinitialiser le zoom
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


# ---------- Utilitaires ----------
def nouvelle_figure(hauteur=450, **kwargs):
    fig = go.Figure(**kwargs)
    _style(fig, hauteur)
    return fig


def _style(fig, hauteur):
    fig.update_layout(
        height=hauteur,
        paper_bgcolor=COULEUR_FOND, plot_bgcolor=COULEUR_FOND,
        font=dict(color=COULEUR_TEXTE),
        margin=dict(l=60, r=20, t=40, b=50),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(bgcolor=COULEUR_PANEL, bordercolor=COULEUR_BORDURE,
                    borderwidth=1, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=COULEUR_GRILLE, zerolinecolor=COULEUR_GRILLE,
                     linecolor=COULEUR_BORDURE)
    fig.update_yaxes(gridcolor=COULEUR_GRILLE, zerolinecolor=COULEUR_GRILLE,
                     linecolor=COULEUR_BORDURE)


def _legende_item(fig, nom, couleur, dash=None, symbole=False):
    """Entree de legende factice (pour les zones/lignes de fond sans trace)."""
    if symbole:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nom,
                                 marker=dict(color=couleur, size=10, symbol="square")))
    else:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nom,
                                 line=dict(color=couleur, dash=dash, width=2)))


def afficher(fig, conteneur=None):
    """Affiche la figure interactive dans Streamlit."""
    (conteneur or st).plotly_chart(fig, use_container_width=True, config=CONFIG)


# ---------- Etape 1 : signal ----------
def fig_signal(df, canaux, debut, duree, fs, crises, journal):
    debut_idx = int(debut * fs)
    fin_idx = int((debut + duree) * fs)
    temps = df["Time"].iloc[debut_idx:fin_idx]

    fig = nouvelle_figure(550)
    brut = {}
    for i, canal in enumerate(canaux):
        signal = df[canal].iloc[debut_idx:fin_idx].values
        brut[canal] = signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        fig.add_trace(go.Scattergl(
            x=temps, y=signal_norm + i * 3, mode="lines", name=canal,
            line=dict(width=1),
            customdata=signal,
            hovertemplate=canal + " : %{customdata:.2f}<extra></extra>"))

    if crises is not None:
        premier = True
        for _, c in crises.iterrows():
            if c["Fin_s"] >= debut and c["Debut_s"] <= debut + duree:
                fig.add_vrect(x0=c["Debut_s"], x1=c["Fin_s"], fillcolor="crimson",
                              opacity=0.25, line_width=0, layer="below")
                if premier:
                    _legende_item(fig, "Crise (annotation)", "crimson", symbole=True)
                    premier = False

    if journal is not None:
        styles = {"artefact": ("gray", "solid", "Artefact (journal)"),
                  "crise": ("darkorange", "dash", "Note crise/absence (journal)")}
        deja = set()
        visibles = journal[(journal["Onset"] >= debut) & (journal["Onset"] <= debut + duree)]
        for _, ev in visibles.iterrows():
            if ev["Type"] not in styles:
                continue
            couleur, trait, nom = styles[ev["Type"]]
            fig.add_vline(x=ev["Onset"], line=dict(color=couleur, dash=trait, width=1.6),
                          opacity=0.8)
            if nom not in deja:
                _legende_item(fig, nom, couleur, dash=trait)
                deja.add(nom)

    fig.update_xaxes(title_text="Temps (secondes)")
    fig.update_yaxes(title_text="Amplitude (normalisee + decalage)", showticklabels=False)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def fig_journal(journal, crises):
    couleurs = {"artefact": "gray", "crise": "crimson", "yeux": "steelblue",
                "hpn": "green", "sli": "purple", "autre": "black"}
    fig = nouvelle_figure(320)
    for k, (typ, coul) in enumerate(couleurs.items()):
        o = journal.loc[journal["Type"] == typ, "Onset"]
        fig.add_trace(go.Scattergl(
            x=o, y=[k] * len(o), mode="markers", name=typ,
            marker=dict(color=coul, size=7),
            hovertemplate=typ + " : %{x:.1f} s<extra></extra>"))
    if crises is not None:
        for _, c in crises.iterrows():
            fig.add_vrect(x0=c["Debut_s"], x1=c["Fin_s"], fillcolor="crimson",
                          opacity=0.2, line_width=0, layer="below")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(couleurs))),
                     ticktext=list(couleurs))
    fig.update_xaxes(title_text="Temps (s)")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ---------- Etape 2 : crises & cible ----------
def fig_pic_spectral(t_spec, puiss, seuil, t_pic):
    fig = nouvelle_figure(380)
    fig.add_trace(go.Scattergl(x=t_spec, y=puiss, mode="lines", name="Puissance 2.5-4 Hz",
                               line=dict(color="crimson", width=1)))
    fig.add_hline(y=seuil, line=dict(color="black", dash="dash", width=1.5))
    fig.add_vline(x=t_pic, line=dict(color="limegreen", dash="dot", width=2))
    _legende_item(fig, "seuil mu+3sigma", "black", dash="dash")
    _legende_item(fig, f"t_pic = {t_pic:.1f} s", "limegreen", dash="dot")
    fig.update_xaxes(title_text="Temps (s)")
    fig.update_yaxes(title_text="Puissance 2.5-4 Hz")
    return fig


def fig_signal_complet(df, canal, crises):
    pas = max(1, len(df) // 200_000)
    sub = df.iloc[::pas]
    fig = nouvelle_figure(420)
    fig.add_trace(go.Scattergl(x=sub["Time"], y=sub[canal], mode="lines", name=canal,
                               line=dict(color=COULEUR_ACCENT, width=0.8)))
    for _, c in crises.iterrows():
        fig.add_vrect(x0=c["Debut_s"], x1=c["Fin_s"], fillcolor="crimson",
                      opacity=0.4, line_width=0, layer="below")
    if not crises.empty:
        _legende_item(fig, "Crise (annotation)", "crimson", symbole=True)
    fig.update_xaxes(title_text="Temps (s)", rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(title_text="Amplitude EEG")
    return fig


# ---------- Etape 3 : features ----------
def fig_comparaison_classes(comp):
    fig = nouvelle_figure(480)
    for col in comp.columns:
        fig.add_trace(go.Bar(x=[str(i) for i in comp.index], y=comp[col], name=str(col)))
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_yaxes(title_text="Valeur moyenne (echelle symlog)", type="log")
    fig.update_xaxes(tickangle=-45)
    return fig


def fig_correlation(dataset, colonnes):
    corr = dataset[colonnes].corr()
    fig = nouvelle_figure(620)
    fig.add_trace(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu_r",
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{y} / %{x} : %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return fig


# ---------- Etape 4 : modeles ----------
def fig_confusion(y_true, y_pred, titre):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Normal", "Crise"]
    fig = nouvelle_figure(320)
    fig.add_trace(go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=16),
        hovertemplate="Reel %{y} / Predit %{x} : %{z}<extra></extra>"))
    fig.update_layout(title=dict(text=titre, font=dict(size=13)), hovermode="closest",
                      margin=dict(l=60, r=10, t=50, b=50))
    fig.update_xaxes(title_text="Predit")
    fig.update_yaxes(title_text="Reel", autorange="reversed")
    return fig


def fig_importances(importances):
    fig = nouvelle_figure(max(460, 38 * importances["Feature"].nunique()))
    for modele, groupe in importances.groupby("Modele", sort=False):
        groupe = groupe.set_index("Feature").reindex(
            importances["Feature"].drop_duplicates())
        ecarts = groupe["Ecart-type"].fillna(0) if "Ecart-type" in groupe else None
        fig.add_trace(go.Bar(
            x=groupe["Importance"].values,
            y=groupe.index.astype(str),
            name=modele,
            orientation="h",
            error_x=dict(type="data", array=ecarts.values, visible=True)
            if ecarts is not None else None,
            hovertemplate=f"{modele}<br>%{{y}} : %{{x:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        hovermode="closest",
        xaxis_title="Baisse du F1 apres permutation",
        yaxis_title="Feature",
    )
    return fig