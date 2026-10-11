"""Data augmentation des fenetres de CRISE (classe 1 uniquement).

Regles : la classe 0 n'est jamais augmentee ; transformations dans le domaine temporel
qui preservent la morphologie des pointes-ondes (~3 Hz) ; les donnees augmentees ne
servent qu'a l'ENTRAINEMENT (jamais au test) ; chaque transformation est documentee
dans un fichier JSON a cote du CSV.
"""
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from eeg_app.config import (AUG_JITTER_MAX_ECH, AUG_SIGMA_MAX, AUG_SNR_MIN_DB,
                            FEATURE_COLS)
from eeg_app.core.data_loader import charger_donnees
from eeg_app.core.features import features_fenetres, normaliser_signal


# ---------- Diagnostic (etape 0) ----------
def diagnostic_desequilibre(dataset):
    """Retourne (n_crise, n_normal, ratio crise/normal, niveau, message)."""
    n_crise = int((dataset["Class"] == 1).sum())
    n_normal = int((dataset["Class"] == 0).sum())
    ratio = n_crise / n_normal if n_normal else 0.0
    if ratio < 1 / 20:
        return n_crise, n_normal, ratio, "indispensable", \
            "Ratio inferieur a 1:20 : l'augmentation est indispensable."
    if ratio > 1 / 5:
        return n_crise, n_normal, ratio, "optionnelle", \
            "Ratio superieur a 1:5 : l'augmentation est optionnelle."
    return n_crise, n_normal, ratio, "recommandee", \
        "Ratio entre 1:20 et 1:5 : l'augmentation est recommandee."


# ---------- Transformations (appliquees aux fenetres de classe 1) ----------
def jitter_temporel(W, max_shift=5, rng=None):
    """Decale chaque fenetre d'un offset aleatoire dans [-max_shift, +max_shift]."""
    rng = rng or np.random.default_rng()
    n, L = W.shape
    out = np.empty_like(W)
    for i in range(n):
        s = int(rng.integers(-max_shift, max_shift + 1))
        if s == 0:
            out[i] = W[i]
        elif s > 0:
            out[i, :-s] = W[i, s:]
            out[i, -s:] = W[i, -1]      # bord : repete la derniere valeur
        else:
            out[i, -s:] = W[i, :s]
            out[i, :-s] = W[i, 0]
    return out


def bruit_gaussien(W, snr_db=25, rng=None):
    """Ajoute un bruit blanc avec un SNR controle (en dB)."""
    rng = rng or np.random.default_rng()
    puissance_signal = np.mean(W ** 2, axis=1, keepdims=True)
    sigma = np.sqrt(puissance_signal / (10 ** (snr_db / 10)))
    return W + rng.standard_normal(W.shape) * sigma


def gain_aleatoire(W, sigma=0.05, rng=None):
    """Applique un gain aleatoire N(1, sigma) par fenetre."""
    rng = rng or np.random.default_rng()
    g = rng.normal(1.0, sigma, size=(W.shape[0], 1))
    return W * g


def time_warp(W, sigma=0.05, rng=None):
    """Warp temporel doux : la duree de la fenetre reste identique."""
    rng = rng or np.random.default_rng()
    n, L = W.shape
    out = np.empty_like(W)
    x_orig = np.arange(L)
    for i in range(n):
        facteur = 1 + rng.normal(0, sigma)
        milieu = float(np.clip(0.5 * facteur, 0.05, 0.95))   # garde x_warp croissant
        x_warp = np.interp(x_orig / (L - 1), [0, 0.5, 1], [0, milieu, 1]) * (L - 1)
        out[i] = np.interp(x_orig, x_warp, W[i])
    return out


METHODES = {
    "jitter_temporel": dict(
        fonction=jitter_temporel, param="max_shift",
        physiologie="L'onset manuel est precis a +-20 ms : la crise peut commencer quelques "
                    "echantillons plus tot ou plus tard."),
    "bruit_gaussien": dict(
        fonction=bruit_gaussien, param="snr_db",
        physiologie="Bruit d'electrodes/amplificateur : 20-30 dB de SNR pour un signal propre."),
    "gain_aleatoire": dict(
        fonction=gain_aleatoire, param="sigma_gain",
        physiologie="L'amplitude absolue varie selon l'impedance et le placement des electrodes "
                    "(+-5 % realiste)."),
    "time_warp": dict(
        fonction=time_warp, param="sigma_warp",
        physiologie="La frequence des pointes-ondes varie de 2.5 a 4 Hz selon le patient et le moment."),
}
_ARG = {"max_shift": "max_shift", "snr_db": "snr_db",
        "sigma_gain": "sigma", "sigma_warp": "sigma"}


def valider_parametres(methodes, params):
    """Refuse les parametres qui feraient perdre la morphologie de la crise."""
    if "bruit_gaussien" in methodes and params["snr_db"] < AUG_SNR_MIN_DB:
        raise ValueError(f"snr_db doit etre >= {AUG_SNR_MIN_DB} dB (sinon la crise est noyee).")
    for cle in ("sigma_gain", "sigma_warp"):
        if params[cle] > AUG_SIGMA_MAX:
            raise ValueError(f"{cle} doit etre <= {AUG_SIGMA_MAX} (au-dela on ne simule plus des crises).")
    if "jitter_temporel" in methodes and not 1 <= params["max_shift"] <= AUG_JITTER_MAX_ECH:
        raise ValueError(f"max_shift doit etre entre 1 et {AUG_JITTER_MAX_ECH} echantillons.")


def cle_dataset(ctx):
    """Identifie le dataset reel auquel une augmentation se rapporte."""
    return (ctx.chemin_fichier, ctx.canal_analyse, ctx.window_sec, ctx.overlap,
            ctx.seuil_crise, ctx.densifier, ctx.exclure_art, ctx.marge_art,
            tuple(ctx.intervalles))


# ---------- Generation ----------
def augmenter_fenetres_crise(chemin, canal, fs, window_sec, dataset, methodes, copies,
                             params, seed=42, normaliser=False):
    """Retourne (aug, resume).

    aug    : une ligne par fenetre augmentee (memes features que le dataset), avec
             Source_window (fenetre reelle d'origine), Augmentation et Copie.
    resume : par methode, nombre de fenetres et variation mediane des features cles.
    """
    valider_parametres(methodes, params)
    cls1 = dataset[dataset["Class"] == 1]            # JAMAIS la classe 0
    if cls1.empty:
        raise ValueError("Aucune fenetre de crise a augmenter.")

    df = charger_donnees(chemin)
    t = df["Time"].values
    x = df[canal].values.astype(float)
    if normaliser:
        x = normaliser_signal(x, canal)
    win = max(8, int(round(window_sec * fs)))
    idx = np.searchsorted(t, cls1["Window_start_s"].values)
    if not np.allclose(t[idx], cls1["Window_start_s"].values):
        raise ValueError("Fenetres introuvables dans le signal (parametres modifies ?).")
    W = sliding_window_view(x, win)[idx]             # copie (n_crise, win)

    rng = np.random.default_rng(seed)
    meta = cls1.drop(columns=FEATURE_COLS + ["Window"]).reset_index(drop=True)
    parent = cls1[FEATURE_COLS].reset_index(drop=True)
    blocs, lignes = [], []
    for nom in methodes:
        m = METHODES[nom]
        kw = {_ARG[m["param"]]: params[m["param"]]}
        for c in range(1, copies + 1):
            Wa = m["fonction"](W, rng=rng, **kw)
            feats = pd.DataFrame(features_fenetres(Wa, fs))
            en_tete = pd.DataFrame({"Source_window": cls1["Window"].values,
                                    "Augmentation": nom, "Copie": c})
            blocs.append(pd.concat([en_tete, feats, meta], axis=1))
            rel = ((feats - parent).abs() / (parent.abs() + 1e-12)).median() * 100
            lignes.append({"Methode": nom, "Copie": c, "Fenetres": len(feats),
                           "Var. mediane RMS (%)": rel["RMS"],
                           "Var. mediane Power_Delta (%)": rel["Power_Delta"],
                           "Var. mediane Power_Theta (%)": rel["Power_Theta"]})
    aug = pd.concat(blocs, ignore_index=True)
    resume = (pd.DataFrame(lignes).groupby("Methode", sort=False)
              .agg({"Fenetres": "sum", "Var. mediane RMS (%)": "mean",
                    "Var. mediane Power_Delta (%)": "mean",
                    "Var. mediane Power_Theta (%)": "mean"}).round(2))
    return aug, resume


# ---------- Sauvegarde (meme repertoire que les autres donnees) ----------
def _slug(texte):
    return re.sub(r"[^A-Za-z0-9_-]+", "", str(texte))


def enregistrer(aug, dossier, patient_id, canal, window_sec, methodes, copies, params,
                seed, dataset, fs, overlap, seuil_crise, densifier):
    """Ecrit le CSV augmente + un JSON de tracabilite. Retourne (chemin_csv, chemin_json)."""
    base = (f"{patient_id}_donnees_augmentees_crises_{_slug(canal)}_"
            f"fenetre{window_sec * 1000:.0f}ms_{'-'.join(m.split('_')[0] for m in methodes)}_"
            f"x{copies}")
    chemin_csv = os.path.join(dossier, base + ".csv")
    chemin_json = os.path.join(dossier, base + "_parametres.json")
    aug.to_csv(chemin_csv, index=False)

    n1 = int((dataset["Class"] == 1).sum())
    n0 = int((dataset["Class"] == 0).sum())
    trace = {
        "description": "Fenetres de CRISE augmentees (classe 1 uniquement). "
                       "A utiliser pour l'entrainement seulement, jamais pour le test.",
        "date": datetime.now().isoformat(timespec="seconds"),
        "patient": patient_id, "canal": canal, "frequence_echantillonnage_hz": round(float(fs), 3),
        "fenetre_s": window_sec, "recouvrement": overlap, "seuil_etiquette_crise": seuil_crise,
        "fenetres_denses_autour_des_crises": densifier,
        "graine_aleatoire": seed, "copies_par_methode": copies,
        "dataset_reel": {"fenetres_crise": n1, "fenetres_normales": n0,
                         "ratio_crise_sur_normal": round(n1 / n0, 5) if n0 else None},
        "fenetres_augmentees": len(aug),
        "classe_0_augmentee": False,
        "transformations": [
            {"nom": nom, "parametre": m["param"], "valeur": params[m["param"]],
             "justification_physiologique": m["physiologie"],
             "fenetres_generees": int((aug["Augmentation"] == nom).sum())}
            for nom, m in ((n, METHODES[n]) for n in methodes)],
        "colonnes_ajoutees": {"Source_window": "numero de la fenetre reelle d'origine",
                              "Augmentation": "transformation appliquee",
                              "Copie": "numero de copie de cette transformation"},
    }
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    return chemin_csv, chemin_json
