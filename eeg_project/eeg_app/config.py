"""Configuration centrale : chemins, constantes metier et couleurs."""
import os

# --- Chemins -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
if not os.path.isfile(os.path.join(IMAGE_DIR, "banner_hero.png")):
    dossier_images_parent = os.path.join(os.path.dirname(BASE_DIR), "image")
    if os.path.isfile(os.path.join(dossier_images_parent, "banner_hero.png")):
        IMAGE_DIR = dossier_images_parent
ASSETS_DIR = os.path.join(BASE_DIR, "eeg_app", "ui", "assets")

DOSSIER_DATA = os.environ.get(
    "EEG_DATA_DIR",
    r"C:\Users\LENOVO\Desktop\Projet_EEG_DataMining\Projet data mining 3AINFO\data_csv",
)

# --- Page Streamlit ----------------------------------------------------
PAGE_TITLE = "Detection Crises EEG"

# --- Constantes metier -------------------------------------------------
SECONDES_PAR_JOUR = 86400
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
FEATURE_COLS = ["Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy",
                "Power_Delta", "Power_Theta", "Power_Alpha", "Power_Beta"]
BANDE_PIC = (2.5, 4)          # bande spectrale typique des pointes-ondes (Hz)
TAILLE_BLOC = 4000            # fenetres traitees par bloc (limite la memoire)
RANDOM_STATE = 42

# --- Annotations (nouveau format Annotations.xlsx) ---------------------
MAX_LIGNES_ANNOT = 2000        # le fichier declare ~1 million de lignes, presque toutes vides
CHAMPS_PATIENT = ["Sexe", "Age", "NombreCE", "TypeAbs", "DebutTrace"]
COLONNES_INTERCRIT = [f"Intercrit{i}" for i in range(1, 8)]
DUREE_MAX_CRISE_S = 120        # au-dela, une crise est consideree comme suspecte
MODE_ALIGN_EXACT = "DebutTrace (exact)"
MODE_ALIGN_PIC = "Pic spectral (manuel)"

# --- Graphiques interactifs -------------------------------------------
MAX_POINTS_COURBE = 100_000    # signal complet : points max affiches
MAX_POINTS_FENETRE = 60_000    # signal detaille : points max par canal
MARGE_CONTEXTE_S = 60          # secondes chargees de part et d'autre pour pouvoir se deplacer

# --- Navigation --------------------------------------------------------
STEPS = ["Signal", "Crises & cible", "Features", "Modeles"]

# --- Couleurs des graphiques (theme clair, turquoise du site) ----------
COULEUR_FOND = "#ffffff"
COULEUR_PANEL = "#ffffff"
COULEUR_BORDURE = "#e3e3e3"
COULEUR_TEXTE = "#1a1a1a"
COULEUR_GRILLE = "#d9d9d9"
COULEUR_ACCENT = "#00a0c6"
# --- Data augmentation (classe crise uniquement) -----------------------
AUG_SNR_MIN_DB = 20            # en dessous, le bruit noie la crise
AUG_SIGMA_MAX = 0.10           # gain et time warp : au-dela on simule d'autres patients / une autre frequence
AUG_JITTER_MAX_ECH = 10        # decalage temporel maximal (echantillons)
AUG_SEED = 42