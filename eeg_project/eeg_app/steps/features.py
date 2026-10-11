"""Etape 3 : dataset de fenetres et analyse des features (mono ou multi-patients)."""
import os

import pandas as pd
import streamlit as st

from eeg_app.config import (AUG_JITTER_MAX_ECH, AUG_SEED, AUG_SIGMA_MAX, AUG_SNR_MIN_DB,
                            DOSSIER_DATA, FEATURE_COLS)
from eeg_app.core.augmentation import (METHODES, augmenter_fenetres_crise, cle_dataset,
                                       diagnostic_desequilibre, enregistrer)
from eeg_app.core.features import moyenne_par_classe
from eeg_app.ui import plots
from eeg_app.ui.components import callout, message_annotations_indisponibles, section


def render(ctx):
    dataset = ctx.get_dataset()
    if dataset is None:
        message_annotations_indisponibles(ctx)
        return

    n1 = _vue_ensemble(ctx, dataset)
    if n1 == 0:
        callout("Aucune fenetre etiquetee 'crise' : les annotations sont probablement mal synchronisees "
                "(voir t_pic dans la barre laterale).", warning=True)
        return

    _analyse_features(dataset)

    # En mode multi, l'augmentation est deja faite dans le pipeline global
    if getattr(ctx, "mode", "mono") == "multi":
        _augmentation_multi_info(ctx, dataset)
    else:
        _augmentation(ctx, dataset)

    st.download_button("Telecharger le dataset (CSV)",
                       dataset.to_csv(index=False).encode("utf-8"),
                       file_name=f"dataset_{ctx.patient_id}.csv", mime="text/csv")


def _analyse_features(dataset):
    st.dataframe(dataset.head(20), use_container_width=True)
    comp = moyenne_par_classe(dataset)
    _comparaison_classes(comp)
    _signification_features()
    plots.afficher(plots.fig_comparaison_classes(comp))
    st.caption("Comparaison des features moyennes entre les deux classes. L'echelle symlog permet de "
               "comparer des valeurs d'ordres de grandeur differents.")
    _correlation(dataset)


def _vue_ensemble(ctx, dataset):
    section("Vue d'ensemble du dataset")
    n1 = int((dataset["Class"] == 1).sum())
    mode = getattr(ctx, "mode", "mono")

    # Nombre de patients (multi) ou simple (mono)
    nb_patients = dataset["Patient_ID"].nunique() if "Patient_ID" in dataset.columns else 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fenetres totales", str(len(dataset)))
    c2.metric("Classe 0 (pas de crise)", str(len(dataset) - n1))
    c3.metric("Classe 1 (crise)", str(n1))
    c4.metric("Ratio crise/normal", f"1 : {(len(dataset) - n1) / n1:.0f}" if n1 else "—")

    # Caption adapte au mode
    if mode == "multi":
        st.caption(
            f"<b>{nb_patients} patient(s)</b>, fenetres de {ctx.window_sec * 1000:.0f} ms, "
            f"recouvrement {ctx.overlap * 100:.0f} %"
            f"{' + pas fin autour des crises' if ctx.densifier else ''}, "
            f"canal {ctx.canal_analyse}, signal standardise separement par enregistrement. "
            "Une fenetre est 'crise' si au moins "
            f"{ctx.seuil_crise * 100:.0f} % de ses points le sont.",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Fenetres de {ctx.window_sec * 1000:.0f} ms, recouvrement {ctx.overlap * 100:.0f} %"
                   f"{' + pas fin autour des crises' if ctx.densifier else ''}, canal {ctx.canal_analyse}. "
                   f"Une fenetre est 'crise' si au moins {ctx.seuil_crise * 100:.0f} % de ses points le sont.")

    # Details par patient (mode multi uniquement)
    if mode == "multi" and "Patient_ID" in dataset.columns:
        _tableau_par_patient(dataset)

    # Info artefacts (mono uniquement, car le journal est par patient)
    if mode == "mono":
        brut = ctx.get_dataset_brut()
        if ctx.journal is not None and brut is not None:
            n_art = int(brut["Artefact"].sum())
            n_art_crise = int(brut.loc[brut["Class"] == 1, "Artefact"].sum())
            st.caption(f"{n_art} fenetres touchent un artefact du journal (dont {n_art_crise} en crise). "
                       + ("Elles sont exclues du dataset." if ctx.exclure_art else
                          "Cochez 'Exclure les fenetres avec artefact' pour les retirer."))
    return n1


def _tableau_par_patient(dataset):
    """Tableau recapitulatif par patient (mode multi)."""
    with st.expander("📊 Details par patient", expanded=False):
        agg = dataset.groupby("Patient_ID").agg(
            Fenetres=("Class", "size"),
            Crises=("Class", "sum"),
        ).reset_index()
        agg["Normales"] = agg["Fenetres"] - agg["Crises"]
        agg["Ratio"] = agg.apply(
            lambda r: f"1 : {r['Normales'] / r['Crises']:.0f}" if r["Crises"] > 0 else "—",
            axis=1)
        st.dataframe(agg, use_container_width=True)
        st.caption("Chaque patient contribue avec son propre nombre de fenetres. "
                   "Un patient avec peu de crises peut tirer le ratio global vers le bas.")


def _comparaison_classes(comp):
    section("Moyenne des features par classe")
    st.dataframe(comp, use_container_width=True)


def _signification_features():
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


def _correlation(dataset):
    section("Matrice de correlation entre features")
    plots.afficher(plots.fig_correlation(dataset, FEATURE_COLS))
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


def _augmentation(ctx, dataset):
    """Mode mono-patient : generation et sauvegarde de l'augmentation."""
    section("Data augmentation des crises")
    n_crise, n_normal, ratio, niveau, msg = diagnostic_desequilibre(dataset)
    callout(f"<b>Diagnostic :</b> {n_crise} fenetres de crise pour {n_normal} normales "
            f"(1 : {n_normal / n_crise:.0f}). {msg}", warning=(niveau == "indispensable"))
    callout("<b>Regles :</b> seule la classe crise est augmentee ; transformations temporelles douces "
            "qui preservent les pointes-ondes (~3 Hz) ; les donnees augmentees ne servent qu'a "
            "l'entrainement (le test reste reel) ; chaque transformation est documentee dans un JSON.")

    noms = list(METHODES)
    methodes = st.multiselect("Transformations", noms, default=noms)
    c1, c2, c3, c4 = st.columns(4)
    copies = c1.number_input("Copies par transformation", 1, 10, 2)
    max_shift = c2.slider("Jitter : decalage max (echantillons)", 1, AUG_JITTER_MAX_ECH, 5,
                          help=f"5 echantillons = {5 / ctx.fs * 1000:.0f} ms a {ctx.fs:.0f} Hz.")
    snr_db = c3.slider("Bruit : SNR (dB)", AUG_SNR_MIN_DB, 40, 25,
                       help=f"Minimum {AUG_SNR_MIN_DB} dB, sinon la crise est noyee.")
    sigma_gain = c4.slider("Gain : sigma", 0.01, AUG_SIGMA_MAX, 0.05, 0.01)
    c5, c6 = st.columns(2)
    sigma_warp = c5.slider("Time warp : sigma", 0.01, AUG_SIGMA_MAX, 0.05, 0.01)
    seed = c6.number_input("Graine aleatoire", 0, 10_000, AUG_SEED)
    params = {"max_shift": max_shift, "snr_db": snr_db,
              "sigma_gain": sigma_gain, "sigma_warp": sigma_warp}

    if st.button("Generer et enregistrer les donnees augmentees", disabled=not methodes):
        try:
            aug, resume = augmenter_fenetres_crise(
                ctx.chemin_fichier, ctx.canal_analyse, ctx.fs, ctx.window_sec, dataset,
                methodes, int(copies), params, int(seed))
            csv, js = enregistrer(
                aug, DOSSIER_DATA, ctx.patient_id, ctx.canal_analyse, ctx.window_sec, methodes,
                int(copies), params, int(seed), dataset, ctx.fs, ctx.overlap, ctx.seuil_crise,
                ctx.densifier)
        except (ValueError, OSError) as e:
            st.error(str(e))
            return
        st.session_state["augmentation"] = {"cle": cle_dataset(ctx), "df": aug,
                                            "resume": resume, "csv": csv, "json": js}

    memo = st.session_state.get("augmentation")
    if memo and memo["cle"] == cle_dataset(ctx):
        callout(f"<b>{len(memo['df'])} fenetres de crise augmentees</b> enregistrees dans le "
                f"repertoire des donnees :<br><code>{os.path.basename(memo['csv'])}</code><br>"
                f"Tracabilite : <code>{os.path.basename(memo['json'])}</code>. "
                "Elles seront ajoutees au train (jamais au test) a l'etape Modeles.")
        st.dataframe(memo["resume"], use_container_width=True)
        st.caption("Variation mediane des features par rapport a la fenetre reelle d'origine : "
                   "elle doit rester faible pour que la morphologie de la crise soit preservee.")


def _augmentation_multi_info(ctx, dataset):
    """Mode multi : l'augmentation a deja ete faite dans le pipeline global."""
    section("Data augmentation (mode multi-patients)")
    n_crise, n_normal, ratio, niveau, msg = diagnostic_desequilibre(dataset)
    callout(f"<b>Diagnostic global :</b> {n_crise} fenetres de crise pour {n_normal} normales "
            f"(1 : {n_normal / n_crise:.0f}). {msg}", warning=(niveau == "indispensable"))

    aug = getattr(ctx, "augmentation", None)
    if aug is not None and len(aug):
        callout(
            f"<b>{len(aug)} fenetres de crise augmentees</b> ont ete generees pour "
            f"l'ensemble des patients selectionnes. Elles seront ajoutees au train "
            "(jamais au test) a l'etape Modeles.<br>"
            "En mode multi, l'augmentation est appliquee <b>automatiquement par patient</b> "
            "avec les parametres par defaut (facteur 2)."
        )
        if "Patient_ID" in aug.columns:
            with st.expander("📊 Augmentation par patient", expanded=False):
                agg = aug.groupby("Patient_ID").size().reset_index(name="Fenetres augmentees")
                st.dataframe(agg, use_container_width=True)
    else:
        callout(
            "Aucune fenetre augmentee n'a ete generee. Verifiez que les parametres du pipeline "
            "multi-patients (dans le module <code>multi_patient.py</code>) sont bien actives.",
            warning=True,
        )