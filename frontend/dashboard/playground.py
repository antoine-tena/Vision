"""Dernière diapositive : dessiner un chiffre, le voir reconnu à côté, recommencer à volonté."""

import streamlit as st
from network.model import DigitClassifier
from pipeline.figures import probabilities_figure
from pipeline.recognition import Recognition, recognize

from dashboard import explanations
from dashboard.canvas import draw_digit
from dashboard.constants import PLAYGROUND_CANVAS_KEY, PLAYGROUND_FIGURE_HEIGHT
from dashboard.explanations import key_fact
from dashboard.formula import Term, show_formula

RESULT_FORMULA_REM = 7
# Un peu au-dessus de 100 % pour laisser la place à l'étiquette de la barre prédite.
PROBABILITY_AXIS_TOP = 115


def render_playground(model: DigitClassifier, recognition: Recognition) -> None:
    """`recognition` (le chiffre des diapositives précédentes) n'est pas utilisé ici."""
    drawing_column, answer_column = st.columns(2, vertical_alignment="center")
    with drawing_column:
        key_fact("Dessinez un chiffre, « Effacer » pour recommencer")
        drawn = draw_digit(PLAYGROUND_CANVAS_KEY)
    with answer_column:
        guess = None if drawn is None else recognize(model, drawn)
        if guess is None or guess.is_blank:
            key_fact("Le chiffre reconnu s'affichera ici")
            return
        show_formula(
            [
                Term(
                    str(guess.digit),
                    explanations.OUTPUT_COLOR,
                    f"chiffre reconnu · confiance {guess.confidence:.0%}",
                )
            ],
            size_rem=RESULT_FORMULA_REM,
        )
        figure = probabilities_figure(guess.probabilities)
        figure.update_yaxes(range=[0, PROBABILITY_AXIS_TOP])
        st.plotly_chart(figure.update_layout(height=PLAYGROUND_FIGURE_HEIGHT))
