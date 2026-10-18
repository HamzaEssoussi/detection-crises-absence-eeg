"""Entrainement et evaluation des classifieurs."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from eeg_app.config import FEATURE_COLS, RANDOM_STATE


class ErreurModelisation(ValueError):
    """Erreur affichable a l'utilisateur (decoupage impossible, une seule classe...)."""


def decouper(X, y, test_size, aleatoire):
    """Decoupage train/test : aleatoire stratifie ou chronologique."""
    try:
        if aleatoire:
            return train_test_split(X, y, test_size=test_size,
                                    random_state=RANDOM_STATE, stratify=y)
        cut = int(len(X) * (1 - test_size))
        return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    except ValueError as e:
        raise ErreurModelisation(f"Decoupage impossible : {e}") from e


def creer_modeles(balanced, modeles_choisis=None):
    """Dictionnaire {nom: (modele, necessite_standardisation)}.

    `modeles_choisis` : liste de noms a inclure. None = tous.
    """
    cw = "balanced" if balanced else None
    tous = {
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE,
                                                 class_weight=cw), False),
        "Random Forest": (RandomForestClassifier(n_estimators=200,
                                                 random_state=RANDOM_STATE,
                                                 class_weight=cw, n_jobs=-1), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True),
        "SVM": (SVC(kernel="rbf", random_state=RANDOM_STATE,
                    class_weight=cw), True),
        "Deep Learning (MLP)": (MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=400,
            random_state=RANDOM_STATE,
        ), True),
    }
    if modeles_choisis is not None:
        tous = {k: v for k, v in tous.items() if k in modeles_choisis}
    return tous


def calculer_metriques(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }


def calculer_importances_permutation(modele, X_test, y_test):
    """Calcule la baisse du F1 quand chaque feature de test est melangee."""
    if y_test.nunique() < 2:
        return None
    resultat = permutation_importance(
        modele, X_test, y_test, scoring="f1",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    return pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": resultat.importances_mean,
        "Ecart-type": resultat.importances_std,
    })


def _entrainer_un_modele(modele, X, y, balanced):
    if balanced and isinstance(modele, MLPClassifier):
        poids = compute_sample_weight(class_weight="balanced", y=y)
        modele.fit(X, y, sample_weight=poids)
    else:
        modele.fit(X, y)


def entrainer_modeles(dataset, test_size, aleatoire, balanced,
                      aug=None, modeles_choisis=None):
    """Entraine les modeles selectionnes et retourne un dictionnaire de resultats.

    `aug` (fenetres de crise augmentees) n'est ajoute qu'a l'ENTRAINEMENT, et
    seulement si la fenetre reelle d'origine est dans le train : le test reste
    100 % reel.

    `modeles_choisis` : liste de noms de modeles a entrainer (None = tous).
    """
    X, y = dataset[FEATURE_COLS], dataset["Class"]
    X_tr, X_te, y_tr, y_te = decouper(X, y, test_size, aleatoire)

    n_aug = 0
    if aug is not None and len(aug):
        parents = set(dataset.loc[X_tr.index, "Window"])
        a = aug[aug["Source_window"].isin(parents)]
        if len(a):
            X_tr = pd.concat([X_tr, a[FEATURE_COLS]], ignore_index=True)
            y_tr = pd.concat([y_tr, a["Class"]], ignore_index=True)
            n_aug = len(a)

    if y_tr.nunique() < 2:
        raise ErreurModelisation(
            "Le jeu d'entrainement ne contient qu'une seule classe avec ce decoupage.")

    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    modeles = creer_modeles(balanced, modeles_choisis)
    if not modeles:
        raise ErreurModelisation("Aucun modele selectionne.")

    lignes, preds, importances = [], {}, []
    for nom, (modele, scale) in modeles.items():
        Xa, Xb = (X_tr_s, X_te_s) if scale else (X_tr, X_te)
        _entrainer_un_modele(modele, Xa, y_tr, balanced)
        y_pred = modele.predict(Xb)
        preds[nom] = y_pred
        lignes.append({"Modele": nom, **calculer_metriques(y_te, y_pred)})
        perm = calculer_importances_permutation(modele, Xb, y_te)
        if perm is not None:
            perm.insert(0, "Modele", nom)
            importances.append(perm)

    return {
        "df": pd.DataFrame(lignes).set_index("Modele").round(3),
        "preds": preds, "y_te": y_te,
        "importances_permutation": pd.concat(importances, ignore_index=True)
        if importances else None,
        "n_train": len(y_tr), "n_test": len(y_te), "n_aug_train": n_aug,
        "crises_train": int(y_tr.sum()), "crises_test": int(y_te.sum()),
    }


def entrainer_modeles_groupkfold(dataset, modeles_choisis=None, balanced=True, aug=None):
    """Entrainement avec validation croisee par patient (GroupKFold).

    Chaque fold a des patients differents dans train et test : c'est la mesure
    la plus honnete de la generalisation inter-patients.

    Retourne un dict compatible avec l'interface existante.
    """
    import numpy as np
    import pandas as pd

    if "Patient_ID" not in dataset.columns:
        raise ErreurModelisation(
            "GroupKFold necessite une colonne 'Patient_ID' dans le dataset.")

    X = dataset[FEATURE_COLS]
    y = dataset["Class"]
    groups = dataset["Patient_ID"]

    n_patients = groups.nunique()
    if n_patients < 2:
        raise ErreurModelisation(
            "GroupKFold necessite au moins 2 patients. Passez en mode mono-patient.")

    n_splits = min(5, n_patients)
    gkf = GroupKFold(n_splits=n_splits)

    modeles = creer_modeles(balanced, modeles_choisis)
    if not modeles:
        raise ErreurModelisation("Aucun modele selectionne.")

    resultats_patient = []
    importances = []
    folds_evalues = 0

    for tr_idx, te_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # Augmentation : uniquement sur le train, uniquement les fenetres du train
        if aug is not None and len(aug):
            patients_train = set(dataset.iloc[tr_idx]["Patient_ID"])
            a = aug[aug["Patient_ID"].isin(patients_train)]
            if len(a):
                X_tr = pd.concat([X_tr, a[FEATURE_COLS]], ignore_index=True)
                y_tr = pd.concat([y_tr, a["Class"]], ignore_index=True)

        if y_tr.nunique() < 2:
            patients_test = sorted(dataset.iloc[te_idx]["Patient_ID"].unique())
            raise ErreurModelisation(
                "Impossible d'entrainer un fold sans les deux classes. "
                f"Patients de test : {', '.join(patients_test)}.")

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        folds_evalues += 1
        patients_test = dataset.iloc[te_idx]["Patient_ID"].to_numpy()

        for nom, (modele, scale) in modeles.items():
            Xa = X_tr_s if scale else X_tr
            Xb = X_te_s if scale else X_te
            _entrainer_un_modele(modele, Xa, y_tr, balanced)
            y_pred = modele.predict(Xb)
            perm = calculer_importances_permutation(modele, Xb, y_te)
            if perm is not None:
                perm.insert(0, "Modele", nom)
                perm.insert(1, "Fold", folds_evalues)
                importances.append(perm)
            for patient in sorted(set(patients_test)):
                masque = patients_test == patient
                metriques = calculer_metriques(y_te.iloc[masque], y_pred[masque])
                resultats_patient.append({
                    "Patient_ID": patient,
                    "Modele": nom,
                    **metriques,
                    "Fenetres test": int(masque.sum()),
                    "Crises test": int(y_te.iloc[masque].sum()),
                })

    if not folds_evalues:
        raise ErreurModelisation("Aucun fold GroupKFold n'a pu etre evalue.")

    df_patient = pd.DataFrame(resultats_patient)
    mesures = ["Accuracy", "Precision", "Recall", "F1-score"]
    resume_patients = df_patient.groupby("Modele")[mesures].mean()
    dispersion_f1 = df_patient.groupby("Modele")["F1-score"].std(ddof=0)
    # Moyennes
    lignes = []
    for nom in modeles:
        lignes.append({
            "Modele": nom,
            **resume_patients.loc[nom].to_dict(),
            "Ecart-type F1 patient": dispersion_f1.loc[nom],
        })

    return {
        "df": pd.DataFrame(lignes).set_index("Modele").round(3),
        "df_patient": df_patient.round(3),
        "preds": {},            # pas de matrice de confusion en mode CV
        "y_te": None,
        "importances_permutation": pd.concat(importances, ignore_index=True)
        if importances else None,
        "n_train": len(X), "n_test": 0,
        "n_aug_train": len(aug) if aug is not None else 0,
        "crises_train": int(y.sum()), "crises_test": 0,
        "mode": "groupkfold",
        "n_patients": n_patients,
    }