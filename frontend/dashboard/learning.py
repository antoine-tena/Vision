"""Partie 2 du diaporama, « Comment il apprend » : la boucle, la perte, la descente de gradient."""

import streamlit as st
from network.model import DigitClassifier
from network.training import TrainingConfig
from pipeline.recognition import Recognition

from dashboard import explanations
from dashboard.explanations import key_fact
from dashboard.formula import show_formula

LOOP = """
- **Au départ** : poids tirés au hasard, le réseau répond n'importe quoi
- **Puis** on répète ces 4 temps des milliers de fois, image après image
- **Le gradient** dit, pour chaque poids, dans quel sens le changer pour moins se tromper
"""

LOSS_TABLE = """
| Probabilité donnée au bon chiffre | 100 % | 87 % | 50 % | 10 % | 1 % |
|---|---|---|---|---|---|
| **Perte L** | 0 | 0,14 | 0,69 | 2,3 | 4,6 |
"""

LOSS_NOTES = """
- **Nulle** si le réseau est parfait ; **énorme** s'il est sûr de lui et se trompe
- Le « - » rend la perte positive ; nom officiel : **entropie croisée**
"""

DESCENT_NOTES = """
- **Image** : dans le brouillard en montagne, on cherche la vallée (l'erreur minimale)
- À chaque pas, on sent **la pente** (le gradient) et on descend un peu
"""


def render_loop_slide(model: DigitClassifier, recognition: Recognition) -> None:
    key_fact("Poids tirés au hasard, puis corrigés des milliers de fois")
    show_formula(explanations.LOOP_FORMULA, size_rem=1.8)
    explanations.explain(LOOP)


def render_loss_slide(model: DigitClassifier, recognition: Recognition) -> None:
    key_fact("La perte : un nombre qui mesure l'erreur")
    show_formula(explanations.LOSS_FORMULA)
    st.markdown(LOSS_TABLE)
    explanations.explain(LOSS_NOTES)


def render_descent_slide(model: DigitClassifier, recognition: Recognition) -> None:
    key_fact("Chaque poids bouge d'un petit pas, dans le sens qui réduit l'erreur")
    show_formula(explanations.DESCENT_FORMULA)
    explanations.explain(DESCENT_NOTES)
    config = TrainingConfig()
    for column, (label, value, meaning) in zip(
        st.columns(3),
        (
            ("Mini-lot", f"{config.batch_size} images", "une correction par lot"),
            ("Époques", str(config.epochs), "passages sur toutes les images"),
            ("Pas η", f"{config.learning_rate}", "puis il diminue"),
        ),
        strict=True,
    ):
        column.metric(label, value)
        column.caption(meaning)
