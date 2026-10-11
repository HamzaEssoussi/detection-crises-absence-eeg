"""Widgets de la barre laterale. Chaque fonction retourne les valeurs choisies."""
import os

import streamlit as st


def titre(texte, separateur=True):
    if separateur:
        st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div class="section-title" style="font-size:1rem;">{texte}</div>',
                        unsafe_allow_html=True)


def choisir_enregistrement(fichiers):
    st.sidebar.markdown('<div class="section-title" style="font-size:1rem;">Parametres</div>',
                        unsafe_allow_html=True)
    return st.sidebar.selectbox("Choisir un enregistrement", fichiers)


def choisir_patients_multi(fichiers):
    st.sidebar.markdown('<div class="section-title" style="font-size:1rem;">Patients</div>',
                        unsafe_allow_html=True)
    return st.sidebar.multiselect(
        "Enregistrements a inclure",
        fichiers,
        default=fichiers,
        help="Selectionnez un ou plusieurs enregistrements EEG a inclure dans le dataset.",
    )


def parametres_affichage(df, canaux_eeg):
    titre("Affichage")
    canaux = st.sidebar.multiselect("Canaux a afficher", canaux_eeg, default=canaux_eeg[:4])
    duree_max = int(df["Time"].iloc[-1])
    debut = st.sidebar.slider("Debut (secondes)", 0, max(duree_max - 1, 1), 0)
    duree = st.sidebar.slider("Duree (secondes)", 1, 30, 5)
    return canaux, debut, duree


def parametres_analyse(canaux_eeg):
    titre("Analyse")
    canal = st.sidebar.selectbox("Canal d'analyse", canaux_eeg)
    fenetre_ms = st.sidebar.slider(
        "Taille de fenetre (ms)", 100, 4000, 500, 50,
        help="Fenetres courtes = meilleure detection des crises tres breves. "
             "Resolution frequentielle ~ 1/duree (250 ms -> 4 Hz).")
    overlap = round(st.sidebar.slider("Recouvrement", 0.0, 0.9, 0.5, 0.1), 2)
    seuil_pct = st.sidebar.slider(
        "Etiquette 'crise' si fenetre en crise a (%)", 5, 100, 30, 5,
        help="Part minimale de la fenetre recouverte par une crise annotee.")
    densifier = st.sidebar.checkbox(
        "Fenetres denses autour des crises", value=True,
        help="Ajoute des fenetres a pas fin (duree/8) autour de chaque crise pour ne "
             "manquer aucune crise courte et reequilibrer les classes.")
    return canal, fenetre_ms / 1000, overlap, seuil_pct / 100, densifier


def champ_annotations(valeur_defaut):
    titre("Annotations")
    return st.sidebar.text_input(
        "Chemin du fichier d'annotations",
        value=valeur_defaut,
        help="Remplacez ce chemin par celui du nouveau fichier .xlsx ou .csv. "
             "Par defaut, l'application cherche Annotations.xlsx puis Annotations.csv "
             "dans le dossier des donnees ou son dossier parent.",
    )


def parametres_pic(fichier, canal, t_pic_auto, duree_totale, crises_brutes):
    """Instant du pic et crise de reference. Retourne (t_pic, ligne_ref)."""
    t_pic = st.sidebar.number_input(
        "Instant du pic de crise dans l'EEG (s)",
        min_value=0.0, max_value=float(duree_totale),
        value=round(t_pic_auto, 1), step=0.5,
        key=f"tpic_{fichier}_{canal}",
        help="Detecte automatiquement (pic 2.5-4 Hz) et suppose coincider avec le debut "
             "de la crise choisie ci-dessous. Ajustez-le si la zone rouge ne colle pas au signal.",
    )
    crise_ref = st.sidebar.selectbox(
        "Crise de l'Excel correspondant au pic", crises_brutes["Crise"].tolist(),
        key=f"cref_{fichier}",
        help="Par defaut la 1ere. Si ce fichier ne contient que les crises suivantes "
             "(ex. fichier _0001), choisissez la crise qui correspond au pic.")
    ligne_ref = crises_brutes[crises_brutes["Crise"] == crise_ref].iloc[0]
    return t_pic, ligne_ref


def parametres_journal(journal, chemin_journal_fichier):
    """Retourne (exclure_art, marge_art)."""
    titre("Journal (fichier h)")
    exclure_art, marge_art = False, 1.0
    if journal is None:
        st.sidebar.caption(f"Pas de journal trouve ({os.path.basename(chemin_journal_fichier)}).")
    else:
        st.sidebar.caption(f"{len(journal)} notes chargees.")
        exclure_art = st.sidebar.checkbox(
            "Exclure les fenetres avec artefact", value=False,
            help="Retire du dataset les fenetres proches d'une note ARTEFACT / BOUGE / SATURATION.")
        marge_art = st.sidebar.slider("Marge autour d'un artefact (s)", 0.0, 5.0, 1.0, 0.5)
    return exclure_art, marge_art


def parametres_journal_multi(nb_journaux, nb_notes):
    """Retourne les reglages d'artefacts communs au mode multi-patients."""
    titre("Journaux (fichiers h)")
    if not nb_journaux:
        st.sidebar.caption("Aucun journal trouve pour les enregistrements selectionnes.")
        return False, 1.0

    st.sidebar.caption(f"{nb_notes} notes chargees dans {nb_journaux} journal(aux).")
    exclure_art = st.sidebar.checkbox(
        "Exclure les fenetres avec artefact", value=False,
        help="Retire les fenetres proches d'une note ARTEFACT / BOUGE / SATURATION.",
    )
    marge_art = st.sidebar.slider("Marge autour d'un artefact (s)", 0.0, 5.0, 1.0, 0.5)
    return exclure_art, marge_art
