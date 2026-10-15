"""Partie 1 du diaporama, « Le projet » : ce qu'est Vision, son architecture, le lien avec les LLM.

Chaque diapositive reçoit le modèle et un chiffre (celui de l'utilisateur ou un exemple), comme
toutes les diapositives, même quand elle n'en a pas besoin.
"""

import streamlit as st
from network.model import DigitClassifier
from pipeline.activity import network_activity
from pipeline.figures import network_figure
from pipeline.recognition import Recognition
from pipeline.sources import DatasetFormatError
from pipeline.transforms import confusion_accuracy

from dashboard import explanations
from dashboard.constants import MISSING_REPORT, SLIDE_FIGURE_HEIGHT
from dashboard.explanations import key_fact
from dashboard.formula import show_formula
from dashboard.loaders import cached_evaluation

PITCH = "Un réseau de neurones écrit à la main en NumPy, qui reconnaît les chiffres"

WHAT_IT_IS = """
- **Apprentissage automatique** : il apprend sur des exemples, personne n'écrit les règles
- **Pas un LLM** : un LLM traite du texte, avec des milliards de paramètres
- **Rien de caché** : propagation, rétropropagation et descente de gradient codées à la main
"""

ARCHITECTURE = """
- **784** : imposé par la taille des images (28 x 28)
- **128** : le seul nombre choisi librement ; plus = plus précis mais plus lent
- **10** : imposé par le nombre de réponses (les chiffres 0 à 9)
"""

LLM_COMPARISON = """
| | **Vision** | **Un LLM** |
|---|---|---|
| Entrée | 784 pixels | du texte découpé en tokens |
| Sortie | 1 chiffre parmi 10 | le mot suivant, parmi ~100 000 |
| Architecture | réseau dense | Transformer (« attention ») |
| Principes | poids, biais, perte, rétropropagation | **les mêmes** |
"""


def render_what_slide(model: DigitClassifier, recognition: Recognition) -> None:
    key_fact(PITCH)
    show_formula(explanations.FAMILY_FORMULA, size_rem=1.7)
    text_column, figures_column = st.columns(2, vertical_alignment="center")
    with text_column:
        explanations.explain(WHAT_IT_IS)
    with figures_column:
        render_figures(model)


def render_figures(model: DigitClassifier) -> None:
    """Les chiffres à connaître par cœur, lus dans le modèle et son évaluation."""
    try:
        evaluation = cached_evaluation()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_REPORT.format(error=error))
        return
    columns = st.columns(len(evaluation) + 1)
    parameters = f"{model.parameter_count:,}".replace(",", " ")
    columns[0].metric("Paramètres", parameters)
    for column, (source, matrix) in zip(columns[1:], evaluation.items(), strict=True):
        column.metric(f"Test, {source}", f"{confusion_accuracy(matrix):.1%}")


def render_architecture_slide(model: DigitClassifier, recognition: Recognition) -> None:
    text_column, figure_column = st.columns([2, 3], vertical_alignment="center")
    with text_column:
        show_formula(explanations.ARCHITECTURE_FORMULA, size_rem=1.8)
        explanations.explain(ARCHITECTURE)
    activity = network_activity(model, recognition.pixels)
    figure = network_figure(activity, activity.neuron_count)
    figure_column.plotly_chart(figure.update_layout(height=SLIDE_FIGURE_HEIGHT))


def render_llm_slide(model: DigitClassifier, recognition: Recognition) -> None:
    key_fact("Un LLM marche sur les mêmes principes, en géant")
    show_formula(explanations.SIZE_FORMULA)
    st.markdown(LLM_COMPARISON)
