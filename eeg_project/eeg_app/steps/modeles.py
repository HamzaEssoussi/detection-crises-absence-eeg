"""Etape 4 : entrainement et comparaison des modeles de classification.

Fonctionne en mode mono-patient et multi-patients.
En multi, GroupKFold par patient est impose pour eviter la fuite.
"""
import pandas as pd
import streamlit as st

from eeg_app.core.augmentation import cle_dataset
from eeg_app.core.model_info import (DESCRIPTIONS, GRILLES_GRID_SEARCH,
                                     HYPERPARAMS_ACTUELS)
from eeg_app.core.modeling import (ErreurModelisation, entrainer_modeles,
                                   entrainer_modeles_groupkfold)
from eeg_app.ui import plots
from eeg_app.ui.components import callout, message_annotations_indisponibles, section


NOMS_MODELES = ["Decision Tree", "Random Forest", "KNN", "SVM"]


def render(ctx):
    dataset = ctx.get_dataset()
    if dataset is None:
        message_annotations_indisponibles(ctx)
        return
    if dataset["Class"].sum() < 2:
        callout("Pas assez de fenetres 'crise' pour entrainer un modele.", warning=True)
        return

    _formulaire_entrainement(ctx, dataset)

    res = st.session_state.get("resultats")
    if res:
        _resultats(res)


def _formulaire_entrainement(ctx, dataset):
    section("Entrainement des modeles")

    mode = getattr(ctx, "mode", "mono")

    # --- Message d'avertissement adapte au mode ---
    if mode == "multi":
        nb_patients = dataset["Patient_ID"].nunique() if "Patient_ID" in dataset.columns else 0
        callout(
            f"<b>Mode multi-patients :</b> {nb_patients} patients, {len(dataset)} fenetres au total. "
            "Le decoupage <b>par patient (GroupKFold)</b> garantit que le modele "
            "est evalue sur des patients qu'il n'a <b>jamais vus</b>, ce qui mesure la vraie "
            "generalisation inter-patients.",
        )
    else:
        callout(
            "Avec un seul enregistrement, les scores sont a prendre avec prudence. Avec un "
            "recouvrement de 50 %, le decoupage aleatoire met des fenetres voisines dans train ET "
            "test (fuite de donnees). Le decoupage chronologique est plus honnete.",
            warning=True,
        )

    # ---- 1. Selection des modeles (avec descriptions depliables) ----
    section("1. Choisir les modeles a entrainer")
    st.caption("Cliquez sur le nom d'un modele pour voir comment il fonctionne.")
    modeles_choisis = []
    cols = st.columns(len(NOMS_MODELES))
    for col, nom in zip(cols, NOMS_MODELES):
        with col:
            if st.checkbox(nom, value=True, key=f"chk_{nom}"):
                modeles_choisis.append(nom)
            with st.expander(f"ℹ️ {nom}", expanded=False):
                st.markdown(DESCRIPTIONS[nom], unsafe_allow_html=True)

    if not modeles_choisis:
        st.session_state.pop("resultats", None)
        st.session_state.pop("resultats_signature", None)
        callout("Selectionnez au moins un modele pour continuer.", warning=True)
        return

    # ---- 2. Parametres d'entrainement ----
    section("2. Parametres d'entrainement")
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Part de test", 0.1, 0.5, 0.2, 0.05)

    # Choix du mode de decoupage selon le contexte
    if mode == "multi" and "Patient_ID" in dataset.columns:
        mode_decoupage = "Par patient (GroupKFold)"
        c2.info("Decoupage par patient obligatoire en mode multi.")
    else:
        options = [
            "Aleatoire stratifie (comme le notebook)",
            "Chronologique (sans fuite)",
        ]
        mode_decoupage = c2.radio("Decoupage", options)

    balanced = c3.checkbox("Ponderer les classes (class_weight='balanced')", value=False,
                           help="Utile car les crises sont rares. Non applicable au KNN.")

    # ---- 3. Augmentation deja calculee ----
    aug = None
    if mode == "multi":
        # En multi, l'augmentation est deja dans ctx
        aug = getattr(ctx, "augmentation", None)
        if aug is not None and len(aug):
            st.info(f"✅ {len(aug)} fenetres augmentees seront ajoutees au train "
                    "(le test reste 100 % reel).")
    else:
        memo = st.session_state.get("augmentation")
        if memo and memo["cle"] == cle_dataset(ctx):
            if st.checkbox(f"Ajouter les {len(memo['df'])} fenetres de crise augmentees a "
                           "l'entrainement (le test reste 100 % reel)", value=True):
                aug = memo["df"]

    signature = _signature_resultats(
        ctx, dataset, modeles_choisis, mode_decoupage, balanced, aug)
    if st.session_state.get("resultats_signature") != signature:
        st.session_state.pop("resultats", None)

    # ---- 4. Hyperparametres utilises (expander) ----
    _afficher_hyperparametres(modeles_choisis, balanced)

    # ---- 5. Entrainement ----
    if st.button("Entrainer les modeles"):
        try:
            with st.spinner("Entrainement en cours..."):
                if mode_decoupage.startswith("Par patient"):
                    resultats = entrainer_modeles_groupkfold(
                        dataset, modeles_choisis=modeles_choisis,
                        balanced=balanced, aug=aug)
                else:
                    resultats = entrainer_modeles(
                        dataset, test_size,
                        aleatoire=mode_decoupage.startswith("Aleatoire"),
                        balanced=balanced, aug=aug,
                        modeles_choisis=modeles_choisis)
        except ErreurModelisation as e:
            st.error(str(e))
            st.stop()
        st.session_state["resultats"] = resultats
        st.session_state["resultats_signature"] = signature


def _signature_resultats(ctx, dataset, modeles, decoupage, balanced, aug):
    """Identifie les reglages et donnees auxquels les scores affiches se rapportent."""
    patients = tuple(sorted(dataset["Patient_ID"].astype(str).unique())) \
        if "Patient_ID" in dataset.columns else ()
    enregistrements = tuple(sorted(dataset["Recording_ID"].astype(str).unique())) \
        if "Recording_ID" in dataset.columns else (ctx.fichier,)
    return (
        getattr(ctx, "mode", "mono"), patients, enregistrements,
        len(dataset), int(dataset["Class"].sum()), ctx.canal_analyse,
        ctx.window_sec, ctx.overlap, ctx.seuil_crise, ctx.densifier,
        ctx.exclure_art, ctx.marge_art, tuple(modeles), decoupage, balanced,
        len(aug) if aug is not None else 0,
    )


def _afficher_hyperparametres(modeles_choisis, balanced):
    """Affiche les hyperparametres utilises + la grille Grid Search recommandee."""
    with st.expander("⚙️ Hyperparametres utilises (et Grid Search recommandee)", expanded=False):
        callout(
            "Les modeles tournent actuellement avec leurs <b>valeurs par defaut</b>. "
            "Pour de meilleurs resultats, utilisez une <b>recherche par grille (GridSearchCV)</b> "
            "avec <b>validation croisee 5-fold</b> et <b>scoring='f1'</b> (adapte au desequilibre). "
            "Les grilles recommandees sont affichees ci-dessous."
        )
        for nom in modeles_choisis:
            st.markdown(f"#### {nom}")
            st.markdown("**Hyperparametres actuels :**")
            st.json(HYPERPARAMS_ACTUELS[nom])
            st.markdown("**Grille Grid Search recommandee :**")
            st.json(GRILLES_GRID_SEARCH[nom])
            if balanced and nom in ("Decision Tree", "Random Forest", "SVM"):
                st.caption("`class_weight` sera fixe a `'balanced'` (case cochee ci-dessus).")
            st.markdown("---")

        st.markdown(
            "**Methode 1 — Grid Search (recommandee) :**"
            "\n\n"
            "```python\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "\n"
            "recherche = GridSearchCV(\n"
            "    modele,\n"
            "    grille,\n"
            "    cv=5,                  # validation croisee 5-fold\n"
            "    scoring='f1',          # metrique adaptee au desequilibre\n"
            "    n_jobs=-1,\n"
            ")\n"
            "recherche.fit(X_train, y_train)\n"
            "print(recherche.best_params_)\n"
            "```\n"
        )


def _resultats(res):
    # Bandeau d'info
    if res.get("mode") == "groupkfold":
        callout(
            "<b>Validation par patient (GroupKFold)</b> : chaque patient de test est "
            "entierement absent de l'apprentissage. Le tableau donne les moyennes macro "
            "par patient et leur dispersion."
        )
    if res.get("n_aug_train"):
        callout(f"<b>{res['n_aug_train']}</b> fenetres de crise augmentees ont ete ajoutees au train "
                "(comprises dans les comptes ci-dessous). Le test ne contient que des donnees reelles.")

    if res.get("mode") != "groupkfold":
        callout(f"<b>Train :</b> {res['n_train']} fenetres ({res['crises_train']} crise) &mdash; "
                f"<b>Test :</b> {res['n_test']} fenetres ({res['crises_test']} crise)")

    st.dataframe(res["df"], use_container_width=True)
    _resume_automatique(res)

    if res.get("mode") == "groupkfold" and "df_patient" in res:
        section("Resultats par patient de test")
        details = res["df_patient"].sort_values(["Patient_ID", "Modele"])
        st.dataframe(details, use_container_width=True)
        st.caption(
            "Chaque ligne mesure le modele sur un patient entier qui n'a contribue "
            "a aucun apprentissage du fold correspondant."
        )

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

    if res["preds"]:
        section("Matrices de confusion")
        cols = st.columns(len(res["preds"]))
        for col, (nom, y_pred) in zip(cols, res["preds"].items()):
            plots.afficher(plots.fig_confusion(res["y_te"], y_pred, nom), conteneur=col)

    if res["preds"]:
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

    if res.get("importances_permutation") is not None:
        section("Importance des features par permutation (tous les modeles)")
        importance = res["importances_permutation"]
        if res.get("mode") == "groupkfold":
            importance = importance.groupby(["Modele", "Feature"], as_index=False).agg(
                Importance=("Importance", "mean"),
                **{"Ecart-type": ("Importance", "std")},
            )
            importance["Ecart-type"] = importance["Ecart-type"].fillna(0)
        plots.afficher(plots.fig_importances(importance))
        st.dataframe(
            importance.sort_values(["Modele", "Importance"], ascending=[True, False]),
            use_container_width=True,
        )
        st.caption(
            "Baisse du F1 quand une feature est melangee sur le jeu de test : plus la baisse "
            "moyenne est elevee, plus le modele depend de cette feature. En validation croisee, "
            "les valeurs sont moyennees sur les folds; l'ecart-type montre leur variabilite. "
            "Une valeur negative ou proche de zero ne prouve pas qu'une feature est inutile."
        )


def _resume_automatique(res):
    """Interprete les scores obtenus et propose des pistes adaptees aux resultats."""
    scores = res["df"].copy()
    if scores.empty or "F1-score" not in scores:
        return

    section("Resume automatique et pistes d'amelioration")
    meilleur_f1 = scores["F1-score"].max()
    ex_aequo = scores[scores["F1-score"] == meilleur_f1]
    meilleur_nom = ex_aequo.sort_values(
        ["Recall", "Precision"], ascending=False).index[0]
    meilleur = scores.loc[meilleur_nom]

    if res.get("mode") == "groupkfold":
        criteres = "F1 macro par patient, puis Recall et Precision pour departager"
    else:
        criteres = "F1-score sur le jeu de test, puis Recall et Precision pour departager"

    callout(
        f"<b>Modele le plus adapte parmi ceux testes : {meilleur_nom}</b> "
        f"(F1 {meilleur['F1-score']:.3f}, Recall {meilleur['Recall']:.3f}, "
        f"Precision {meilleur['Precision']:.3f}). "
        f"Classement base sur {criteres}. Ce resultat ne vaut que pour les donnees et "
        "parametres de cette execution; il ne garantit pas une superiorite sur d'autres patients."
    )

    conseils = []
    for nom, scores_modele in scores.iterrows():
        rappel = float(scores_modele["Recall"])
        precision = float(scores_modele["Precision"])
        f1 = float(scores_modele["F1-score"])
        ecart = meilleur_f1 - f1
        if rappel < precision:
            point_faible = (
                f"Recall plus faible que la precision ({rappel:.3f} contre {precision:.3f}) : "
                "le modele manque davantage de crises."
            )
            action = "Tester class_weight='balanced' et choisir le seuil avec une validation croisee groupee, en privilegiant le Recall."
        elif precision < rappel:
            point_faible = (
                f"Precision plus faible que le Recall ({precision:.3f} contre {rappel:.3f}) : "
                "le modele genere relativement plus de fausses alertes."
            )
            action = "Tester un seuil de decision plus exigeant et comparer le compromis Precision-Recall sur les folds de validation."
        else:
            point_faible = (
                f"Precision et Recall sont proches ({precision:.3f} et {rappel:.3f})."
            )
            action = "Explorer les hyperparametres avec une recherche imbriquee et groupee, puis confirmer sur des patients jamais vus."

        if nom == "Decision Tree":
            action += " Pour l'arbre, essayer notamment max_depth et min_samples_leaf."
        elif nom == "Random Forest":
            action += " Pour la foret, essayer max_depth, min_samples_leaf et n_estimators."
        elif nom == "KNN":
            action += " Pour KNN, comparer n_neighbors, weights et metric."
        elif nom == "SVM":
            action += " Pour le SVM, ajuster C, gamma et class_weight."

        if scores_modele["Accuracy"] - f1 > 0.2:
            action += " L'Accuracy depasse nettement le F1 : ne pas optimiser l'Accuracy seule, le desequilibre des classes peut la rendre trompeuse."
        if ecart == 0:
            verdict = "Meilleur F1 observe (ex aequo possible)."
        else:
            verdict = f"F1 inferieur de {ecart:.3f} au meilleur observe."
        conseils.append({
            "Modele": nom,
            "Lecture des scores": f"{verdict} {point_faible}",
            "Piste a tester": action,
        })

    st.dataframe(pd.DataFrame(conseils).set_index("Modele"), use_container_width=True)
    if res.get("mode") != "groupkfold":
        positives = res.get("crises_test", 0)
        callout(
            f"Evaluation sur un seul decoupage : {positives} fenetres de crise dans le test. "
            "Le modele classe premier est seulement le meilleur de cet essai; utilisez une "
            "validation groupee et plusieurs patients pour conclure sur la generalisation.",
            warning=True,
        )
    else:
        details = res.get("df_patient")
        if details is not None and not details.empty:
            pire = details.groupby("Patient_ID")["F1-score"].mean().idxmin()
            score_pire = details.groupby("Patient_ID")["F1-score"].mean().min()
            positives = details.groupby("Patient_ID")["Crises test"].first()
            sans_crise = int((positives == 0).sum())
            note = (
                f"F1 moyen le plus faible : patient {pire} ({score_pire:.3f})."
            )
            if sans_crise:
                note += (
                    f" {sans_crise} patient(s) n'ont aucune fenetre de crise dans leur test; "
                    "leurs scores doivent etre interpretes avec cette limite."
                )
            st.caption(note)