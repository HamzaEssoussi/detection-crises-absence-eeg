"""Etapes (vues) de l'application. Chaque module expose une fonction render(ctx)."""
from eeg_app.steps import crises, features, modeles, signal

# Nom de l'etape (voir config.STEPS) -> fonction d'affichage
ETAPES = {
    "Signal": signal.render,
    "Crises & cible": crises.render,
    "Features": features.render,
    "Modeles": modeles.render,
}
