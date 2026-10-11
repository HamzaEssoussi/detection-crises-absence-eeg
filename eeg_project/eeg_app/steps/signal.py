"""Etape 1 : visualisation du signal EEG, statistiques et journal du technicien."""
import streamlit as st

from eeg_app.ui import plots
from eeg_app.ui.components import callout, legend_item, section


def render(ctx):
    if getattr(ctx, "mode", "mono") == "multi":
        callout(
            "La vue du signal est disponible pour un seul enregistrement. "
            "Desactivez le mode multi-patients pour inspecter un signal individuel. "
            "Le dataset fusionne et les details par patient sont disponibles dans l'etape Features."
        )
        return

    _signal(ctx)
    _statistiques(ctx)
    if ctx.journal is not None:
        _journal(ctx)


def _signal(ctx):
    debut, duree, journal = ctx.debut, ctx.duree, ctx.journal
    section("Signal EEG", f"{debut}s a {debut + duree}s")

    if not ctx.canaux_selectionnes:
        callout("Selectionnez au moins un canal.", warning=True)
        return

    fig = plots.fig_signal(ctx.df, ctx.canaux_selectionnes, debut, duree, ctx.fs,
                           ctx.crises, journal)
    plots.afficher(fig)

    section("Legende du graphique")
    legend_item("#00a0c6", "Courbes", "Chaque couleur represente un canal EEG (electrode). Decalees verticalement.")
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


def _statistiques(ctx):
    section("Statistiques descriptives")
    if not ctx.canaux_selectionnes:
        return
    st.dataframe(ctx.df[ctx.canaux_selectionnes].describe().T, use_container_width=True)
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


def _journal(ctx):
    journal = ctx.journal
    section("Journal du technicien (fichier h)")
    plots.afficher(plots.fig_journal(journal, ctx.crises))

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
