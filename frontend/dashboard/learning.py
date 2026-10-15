"""Page « Comment il apprend » : la perte, la rétropropagation et la descente de gradient."""

import streamlit as st
from network.training import TrainingConfig
from pipeline.sources import DatasetFormatError

from dashboard import explanations
from dashboard.constants import LEARNING_TITLE, MISSING_REPORT
from dashboard.explanations import key_fact
from dashboard.formula import show_formula
from dashboard.loaders import cached_history

LOOP = """
1. **Propagation avant** : on montre une image, il fait une prédiction
2. **Perte** : un nombre qui mesure son erreur
3. **Rétropropagation** : pour chaque poids, dans quel sens le changer (le **gradient**)
4. **Descente de gradient** : chaque poids bouge d'un petit pas dans ce sens
"""

LOSS_TABLE = """
| Probabilité donnée au bon chiffre | 100 % | 87 % | 50 % | 10 % | 1 % |
|---|---|---|---|---|---|
| **Perte L** | 0 | 0,14 | 0,69 | 2,3 | 4,6 |
"""

LOSS_NOTES = """
- Nulle si le réseau est parfait ; énorme s'il est sûr de lui **et** se trompe
- Le « - » rend la perte positive (le log d'un nombre entre 0 et 1 est négatif)
- Nom officiel : **entropie croisée**
"""

DESCENT_NOTES = """
- Image : en pleine montagne dans le brouillard, on cherche la vallée (l'erreur minimale)
- À chaque pas, on sent la pente sous ses pieds (le **gradient**) et on descend un peu
"""


def render_learning() -> None:
    st.title(LEARNING_TITLE)
    key_fact("Poids tirés au hasard, puis corrigés des milliers de fois")
    st.markdown(LOOP)
    st.subheader("La perte : à quel point il se trompe")
    show_formula(explanations.LOSS_FORMULA)
    st.markdown(LOSS_TABLE)
    st.markdown(LOSS_NOTES)
    st.subheader("La descente de gradient : corriger chaque poids")
    show_formula(explanations.DESCENT_FORMULA)
    st.markdown(DESCENT_NOTES)
    render_rhythm()


def render_rhythm() -> None:
    st.subheader("Le rythme de l'entraînement")
    config = TrainingConfig()
    batch, epochs, rate, loss_drop = st.columns(4)
    batch.metric("Mini-lot", f"{config.batch_size} images", "une correction par lot")
    epochs.metric("Époques", config.epochs, "passages sur toutes les images")
    rate.metric("Pas η", f"{config.learning_rate}", "puis il diminue")
    try:
        history = cached_history()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_REPORT.format(error=error))
        return
    loss_drop.metric(
        "Perte", f"{history[-1].loss:.3f}", f"{history[-1].loss - history[0].loss:+.3f}"
    )
