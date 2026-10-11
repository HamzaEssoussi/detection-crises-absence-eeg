"""Etape 2 : crises annotees, detection du pic spectral et variable cible."""
import numpy as np
import streamlit as st

from eeg_app.core.annotations import repartition_cible, verifier_avec_journal
from eeg_app.ui import plots
from eeg_app.ui.components import callout, message_annotations_indisponibles, section


def render(ctx):
    if getattr(ctx, "mode", "mono") == "multi":
        callout(
            "La verification visuelle des crises est disponible pour un seul enregistrement. "
            "Desactivez le mode multi-patients pour inspecter l'alignement d'un patient. "
            "Les fenetres fusionnees sont disponibles dans l'etape Features."
        )
        return

    if ctx.crises is None:
        message_annotations_indisponibles(ctx)
        return

    _crises_brutes(ctx)
    _detection_spectrale(ctx)
    _crises_converties(ctx)
    _verification_journal(ctx)
    _signal_complet(ctx)
    _variable_cible(ctx)


def _crises_brutes(ctx):
    section("Crises annotees", "Valeurs Excel brutes = fraction de jour")
    st.dataframe(ctx.crises_brutes, use_container_width=True)
    callout(
        "<b>Signification des colonnes :</b>"
        "<ul>"
        "<li><b>Crise</b> : numero de la crise (1, 2, 3...).</li>"
        "<li><b>Debut_brut</b> : heure de debut au format Excel (fraction de jour, ex. 0.5 = 12h00).</li>"
        "<li><b>Fin_brut</b> : heure de fin au format Excel.</li>"
        "</ul>"
    )


def _detection_spectrale(ctx):
    section("Detection spectrale", f"Puissance 2.5-4 Hz ({ctx.canal_analyse})")
    plots.afficher(plots.fig_pic_spectral(ctx.t_spec, ctx.puiss, ctx.seuil, ctx.t_pic))
    callout(
        "<b>Comprendre le spectre de puissance :</b>"
        "<ul>"
        "<li>Courbe rouge : puissance du signal dans la bande 2.5-4 Hz, typique des crises d'absence "
        "(pointes-ondes a environ 3 Hz).</li>"
        "<li>Ligne noire pointillee (seuil mu+3sigma) : seuil statistique = moyenne + 3 ecarts-types.</li>"
        "<li>Ligne verte pointillee (t_pic) : instant du pic maximum de puissance 2.5-4 Hz, point d'ancrage "
        "pour synchroniser les annotations.</li>"
        f"<li><b>Ratio pic/seuil actuel :</b> {ctx.puiss.max() / ctx.seuil:.2f} &mdash; un ratio superieur a 1 indique "
        "une activite 2.5-4 Hz nettement au-dessus du bruit de fond.</li>"
        "</ul>"
    )


def _crises_converties(ctx):
    section("Crises converties en temps relatif au debut de l'EEG")
    affichage = ctx.crises_toutes[["Crise", "Debut_s", "Fin_s", "Duree_s"]].round(2)
    affichage["Dans ce fichier"] = np.where(
        ctx.crises_toutes["Crise"].isin(ctx.crises["Crise"]), "Oui", "Non")
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

    if ctx.crises.empty:
        callout("Aucune crise de l'Excel ne tombe dans cet enregistrement. Verifiez t_pic et la crise "
                "choisie dans la barre laterale.", warning=True)
    else:
        callout(f"<b>{len(ctx.crises)} crise(s) sur {len(ctx.crises_toutes)}</b> sont dans cet enregistrement.")


def _verification_journal(ctx):
    section("Verification avec le journal du technicien")
    if ctx.journal is None:
        st.caption("Aucun fichier h trouve pour cet enregistrement.")
        return

    verif, art, notes = verifier_avec_journal(ctx.crises, ctx.journal)
    st.dataframe(verif.round(2), use_container_width=True)

    if len(art):
        d_art = float(np.min(np.abs(art - ctx.t_pic)))
        if d_art <= 3:
            callout(f"Le pic (t = {ctx.t_pic:.1f} s) est a {d_art:.1f} s d'une note ARTEFACT/BOUGE : "
                    "la detection est peut-etre tombee sur un mouvement et non sur la crise. "
                    "Verifiez l'alignement.", warning=True)
        else:
            callout(f"Le pic est a {d_art:.0f} s de l'artefact le plus proche : pas de confusion apparente.")
    if len(notes):
        st.caption("Notes crise/absence du technicien : " + ", ".join(f"{n:.0f} s" for n in notes))
    else:
        st.caption("Le technicien n'a note aucune crise/absence dans ce journal.")


def _signal_complet(ctx):
    section("Signal complet avec crises positionnees")
    plots.afficher(plots.fig_signal_complet(ctx.df, ctx.canal_analyse, ctx.crises))
    callout(
        "Courbe bleue : signal EEG complet (sous-echantillonne pour l'affichage). "
        "Zones rouges translucides : periodes de crise annotees, positionnees sur le signal. "
        "<b>Objectif :</b> verifier visuellement que les zones rouges correspondent bien a des bouffees "
        "d'activite anormale sur le signal."
    )


def _variable_cible(ctx):
    rep = repartition_cible(ctx.df, ctx.intervalles)
    c1, c2, c3 = st.columns(3)
    c1.metric("Echantillons hors crise (0)", str(rep[0]))
    c2.metric("Echantillons en crise (1)", str(rep[1]))
    c3.metric("Proportion de crise", f"{rep[1] / len(ctx.df) * 100:.3f} %")
    callout(
        "<b>Variable cible (Class) :</b> 0 = echantillon hors crise (etat normal) ; "
        "1 = echantillon pendant une crise. La proportion de crise est tres faible (moins de 1 %) : "
        "c'est un probleme de classification desequilibree."
    )
