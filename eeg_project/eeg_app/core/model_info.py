"""Fiches descriptives des modeles : fonctionnement, hyperparametres, grilles Grid Search."""

# --- Descriptions courtes (affichees en expander) ---
DESCRIPTIONS = {
    "Decision Tree": (
        "Un <b>arbre de decision</b> pose une suite de questions binaires sur les features "
        "(ex. <i>Power_Delta &gt; 0.35 ?</i>) et predit la classe au bout de l'arbre. "
        "Chaque noeud choisit la feature qui separe le mieux les crises des normaux "
        "(critere de Gini). <b>Tres interpretable</b> mais <b>surapprend</b> facilement "
        "si on ne limite pas la profondeur (<code>max_depth</code>)."
    ),
    "Random Forest": (
        "Une <b>foret aleatoire</b> = plusieurs centaines d'arbres de decision entraines "
        "chacun sur un echantillon aleatoire des donnees et des features. La prediction finale "
        "est un <b>vote majoritaire</b>. <b>Robuste</b>, resiste au surapprentissage, et fournit "
        "une mesure d'<b>importance des features</b> tres utile pour comprendre quelles "
        "caracteristiques discriminent les crises."
    ),
    "KNN": (
        "Le <b>k plus proches voisins</b> classe une fenetre en regardant ses <code>k</code> "
        "voisins les plus proches dans l'espace des features normalisees. Si la majorite des "
        "voisins sont des crises, la fenetre est classee crise. <b>Simple, non-parametrique</b>, "
        "mais <b>sensible a l'echelle</b> (d'ou la standardisation) et <b>lent</b> sur de gros "
        "datasets (il doit calculer toutes les distances)."
    ),
    "SVM": (
        "Une <b>machine a vecteurs de support</b> cherche l'<b>hyperplan optimal</b> qui separe "
        "les deux classes avec la plus grande marge possible. Le noyau <b>RBF</b> projette les "
        "donnees dans un espace de dimension superieure pour gerer les frontieres non lineaires. "
        "<b>Efficace en haute dimension</b>, mais <b>sensible aux hyperparametres</b> C et gamma, "
        "et difficile a interpreter."
    ),
    "Deep Learning (MLP)": (
        "Un <b>perceptron multicouche (MLP)</b> apprend des representations non lineaires par "
        "retropropagation du gradient. Ce reseau utilise deux couches cachees de 64 et 32 neurones "
        "avec une activation ReLU et l'optimiseur Adam. Les features sont standardisees avant "
        "l'apprentissage; l'option d'equilibrage applique des poids aux exemples. "
        "<b>Flexible</b>, mais sensible au volume de donnees et au surapprentissage."
    ),
}

# --- Hyperparametres utilises actuellement (valeurs par defaut) ---
HYPERPARAMS_ACTUELS = {
    "Decision Tree": {
        "max_depth": "None (illimite)",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini",
        "class_weight": "balanced ou None",
        "random_state": 42,
    },
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": "None (illimite)",
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced ou None",
        "random_state": 42,
    },
    "KNN": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "euclidean (minkowski p=2)",
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced ou None",
        "random_state": 42,
    },
    "Deep Learning (MLP)": {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate_init": 0.001,
        "max_iter": 400,
        "class_weight": "balanced via sample_weight, ou None",
        "random_state": 42,
    },
}

# --- Grilles Grid Search recommandees (pour aller plus loin) ---
GRILLES_GRID_SEARCH = {
    "Decision Tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", None],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", None],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf", "linear"],
        "class_weight": ["balanced", None],
    },
    "Deep Learning (MLP)": {
        "hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.0001, 0.001],
    },
}