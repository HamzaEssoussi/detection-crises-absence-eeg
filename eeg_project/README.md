# Detection automatique des crises d'absence - EEG

Application Streamlit : visualisation du signal, alignement des crises annotees,
extraction de features par fenetre et classification (Decision Tree, Random Forest, KNN, SVM).

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dossier des donnees : variable d'environnement `EEG_DATA_DIR`
(sinon la valeur par defaut dans `eeg_app/config.py`).

### Choisir le fichier d'annotations

Deposez le nouveau fichier dans le dossier des donnees (`EEG_DATA_DIR`) et nommez-le
`Annotations.xlsx` pour qu'il soit detecte automatiquement (ou `Annotations.csv`).
Sinon, dans Streamlit, modifiez **Chemin du fichier d'annotations** dans la barre
laterale et indiquez le chemin complet du fichier. Le champ existe en mode mono et
multi-patients. Le chargeur relit le fichier lorsqu'il est modifie, meme si son nom
reste le meme.

## Mode multi-patients

Cochez **Mode multi-patients** dans la barre laterale, selectionnez les enregistrements
a fusionner, puis choisissez un fichier d'annotations valide et un canal EEG present
dans tous les enregistrements selectionnes. Les etapes **Features** et **Modeles**
exploitent le dataset fusionne ; les vues de signal et d'alignement des crises restent
disponibles en mode mono-patient. Les journaux associes peuvent etre utilises pour
exclure les fenetres proches des artefacts.

En mode multi, les fichiers suffixes `_0000` et `_0001` sont rattaches au meme ID
patient utilise par les annotations. Les enregistrements portant le meme ID d'annotation
sont aussi regroupes pour la validation croisee. Le canal commun choisi est standardise
separement pour chaque enregistrement (moyenne 0, ecart-type 1), sans utiliser les
etiquettes de crise. Cette normalisation utilise le signal complet de l'enregistrement :
elle convient a une analyse qui dispose du fichier EEG entier, mais n'est pas une
evaluation strictement en ligne.

Les durees d'annotation superieures a `DUREE_MAX_CRISE_S` (120 s par defaut), les
horaires invalides et les paires debut/fin incompletes font exclure l'enregistrement
du pool, avec un motif affiche. Les annotations dupliquees pour un meme ID sont
regroupees et dedupliquees. Le mode multi impose GroupKFold par patient et affiche les
metriques par patient en plus du resume macro. Les performances doivent etre lues
patient par patient : une moyenne globale ne garantit pas une performance homogene.
Les colonnes `DebutCE1`/`FinCE1` (et les suivantes) acceptent les fractions de jour de
l'ancien fichier ou les heures `HH:MM:SS` du fichier mis a jour. Si `DebutTrace` est
present et renseigne, il sert d'ancre temporelle exacte pour positionner les crises
depuis le debut de l'EEG; sans cette colonne, le mode mono conserve l'alignement
spectral manuel et le mode multi utilise le pic spectral.
Apres l'entrainement, l'etape **Modeles** classe uniquement les modeles selectionnes
selon les scores de l'execution courante (F1, puis Recall et Precision) et genere pour
chacun une interpretation des forces/faiblesses observees ainsi que des pistes de
reglage. Ces pistes sont des essais a valider, pas des ameliorations garanties. Les
resultats sont masques lorsque les parametres ou le dataset changent afin d'eviter de
presenter des scores d'une execution precedente.

L'importance des features est estimee pour chaque modele par permutation sur le jeu
de test : elle mesure la variation du F1 lorsque les valeurs d'une feature sont
melangees. En validation croisee, les importances sont moyennees entre les folds et
leur ecart-type est affiche. Cette mesure est une indication de dependance du modele,
pas une preuve de causalite; les features correlees peuvent partager ou masquer leur
importance.

## Architecture

```
projet/
├── app.py                     # point d'entree (40 lignes) : assemble les pieces
├── requirements.txt
├── image/                     # banner_hero, fonctions, composants, processus, faits (.png)
└── eeg_app/
    ├── config.py              # chemins, constantes (bandes, features, couleurs, etapes)
    ├── core/                  # LOGIQUE METIER (aucun affichage)
    │   ├── data_loader.py     # lecture EEG / annotations / journal
    │   ├── annotations.py     # extraction + alignement des crises, verifications
    │   ├── spectral.py        # detection du pic 2.5-4 Hz
    │   ├── features.py        # fenetrage + features temps/frequence
    │   ├── modeling.py        # decoupage, entrainement, metriques
    │   └── context.py         # AppContext : etat partage entre les etapes
    ├── ui/                    # INTERFACE reutilisable
    │   ├── assets/theme.css   # tout le CSS
    │   ├── theme.py           # chargement du theme, sidebar ouverte
    │   ├── components.py      # section, callout, legend_item, messages
    │   ├── plots.py           # toutes les figures matplotlib
    │   ├── sidebar.py         # widgets de la barre laterale
    │   ├── context_builder.py # sidebar + core  ->  AppContext
    │   └── layout.py          # barre d'etapes, bandeau, resume, pied de page
    └── steps/                 # UNE VUE PAR ETAPE (fonction render(ctx))
        ├── signal.py
        ├── crises.py
        ├── features.py
        └── modeles.py
```

**Regles de dependance** : `steps` -> `ui` + `core` ; `ui` -> `core` ; `core` ne depend jamais de `ui`/`steps`.

## Ajouter une etape
1. Ajouter son nom dans `STEPS` (`config.py`)
2. Creer `eeg_app/steps/mon_etape.py` avec `render(ctx)`
3. L'enregistrer dans le dictionnaire `ETAPES` (`steps/__init__.py`)



pip install streamlit pandas numpy scipy matplotlib seaborn scikit-learn openpyxl

streamlit run app.py