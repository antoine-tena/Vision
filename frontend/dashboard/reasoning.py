"""Les diapositives du raisonnement, de l'image reçue jusqu'au chiffre reconnu.

Chaque diapositive tient sur un écran : le texte à gauche, la figure ou la formule à droite.
"""

import numpy as np
import streamlit as st
from network.model import DIGIT_COUNT, PIXEL_COUNT, DigitClassifier
from pipeline.figures import (
    evidence_figure,
    hidden_patterns_figure,
    neurons_figure,
    patterns_figure,
    probabilities_figure,
)
from pipeline.reasoning import (
    Reasoning,
    evidence_totals,
    explain,
    hidden_patterns_mosaic,
    neuron_patterns,
)
from pipeline.recognition import Recognition
from pipeline.transforms import digit_image

from dashboard import explanations
from dashboard.constants import (
    PATTERN_GRID_COLUMNS,
    SLIDE_FIGURE_HEIGHT,
    SLIDE_IMAGE_WIDTH,
    TOP_SHAPES,
)
from dashboard.explanations import key_fact
from dashboard.formula import Term, show_formula

TEXT_SHARE = (2, 3)
NARROW_TEXT_SHARE = (1, 2)
SMALL_FORMULA_REM = 1.6
RESULT_FORMULA_REM = 8


def render_input_step(model: DigitClassifier, recognition: Recognition) -> None:
    text_column, image_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        key_fact(f"28 x 28 = {PIXEL_COUNT} pixels → {PIXEL_COUNT} nombres entre 0 et 1")
        show_formula(explanations.INPUT_FORMULA, size_rem=SMALL_FORMULA_REM)
        explanations.explain(explanations.INPUT)
    image_column.image(
        digit_image(recognition.pixels), caption="L'image en 28 x 28", width=SLIDE_IMAGE_WIDTH
    )


def render_shapes_step(model: DigitClassifier, recognition: Recognition) -> None:
    reasoning = explain(model, recognition.pixels, recognition.digit)
    shapes = reasoning.active_shapes[:TOP_SHAPES]
    text_column, figure_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        key_fact(f"{reasoning.active_count} neurones allumés sur {reasoning.hidden_size}")
        explanations.explain("\n".join(f"- **{count} x** {shape}" for shape, count, _ in shapes))
        explanations.explain(explanations.SHAPES)
    examples = neuron_patterns(model, np.array([neuron for _, _, neuron in shapes]))
    titles = [f"{shape}<br>(n°{neuron})" for shape, _, neuron in shapes]
    figure = patterns_figure(examples, titles, "Un neurone exemple par motif : ses poids")
    figure_column.plotly_chart(figure.update_layout(height=SLIDE_FIGURE_HEIGHT))


def render_neuron_step(model: DigitClassifier, recognition: Recognition) -> None:
    key_fact("Chaque neurone calcule une note, puis s'allume si elle est positive")
    show_formula(explanations.NEURON_FORMULA)
    formula_column, text_column = st.columns(2)
    with formula_column:
        show_formula(explanations.RELU_FORMULA)
    with text_column:
        explanations.explain(explanations.WEIGHT_SIGNS)


def render_emergence_step(model: DigitClassifier, recognition: Recognition) -> None:
    text_column, figure_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        key_fact("Personne ne programme les motifs : ils émergent de l'entraînement")
        explanations.explain(explanations.EMERGENCE)
    mosaic = hidden_patterns_mosaic(model, PATTERN_GRID_COLUMNS)
    figure = hidden_patterns_figure(mosaic).update_layout(height=SLIDE_FIGURE_HEIGHT)
    figure_column.plotly_chart(figure)


def chosen_reasoning(model: DigitClassifier, recognition: Recognition, key: str) -> Reasoning:
    # Le choix revient sur le chiffre reconnu quand l'image change (index par défaut).
    digit = st.selectbox(
        "Chiffre expliqué",
        range(DIGIT_COUNT),
        index=recognition.digit,
        key=key,
        help="Par défaut le chiffre reconnu ; un autre chiffre montre ce qui l'a écarté.",
    )
    return explain(model, recognition.pixels, digit)


def render_vote_step(model: DigitClassifier, recognition: Recognition) -> None:
    choice_column, fact_column = st.columns(NARROW_TEXT_SHARE, vertical_alignment="bottom")
    with choice_column:
        reasoning = chosen_reasoning(model, recognition, "chiffre-votes")
    with fact_column:
        key_fact(f"Score du {reasoning.digit} = somme des votes = {reasoning.score:+.2f}")
    text_column, figure_column = st.columns(NARROW_TEXT_SHARE, vertical_alignment="center")
    with text_column:
        show_formula(explanations.VOTE_FORMULA, size_rem=SMALL_FORMULA_REM)
        explanations.explain(explanations.NEURON_VOTES)
    figure = neurons_figure(reasoning).update_layout(height=SLIDE_FIGURE_HEIGHT)
    figure_column.plotly_chart(figure)


def render_evidence_step(model: DigitClassifier, recognition: Recognition) -> None:
    text_column, figure_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        reasoning = chosen_reasoning(model, recognition, "chiffre-pixels")
        support, against = evidence_totals(reasoning.evidence)
        key_fact(f"Pixels pour le {reasoning.digit} : +{support:.1f} · contre : -{against:.1f}")
        show_formula(explanations.EVIDENCE_FORMULA, size_rem=SMALL_FORMULA_REM)
        explanations.explain(explanations.EVIDENCE)
    figure = evidence_figure(reasoning.evidence, reasoning.digit)
    figure_column.plotly_chart(figure.update_layout(height=SLIDE_FIGURE_HEIGHT))


def render_softmax_step(model: DigitClassifier, recognition: Recognition) -> None:
    text_column, figure_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        show_formula(explanations.SOFTMAX_FORMULA, size_rem=SMALL_FORMULA_REM)
        explanations.explain(explanations.PROBABILITIES)
    with figure_column:
        key_fact(f"{recognition.digit} : {recognition.confidence:.0%} des chances")
        figure = probabilities_figure(recognition.probabilities)
        st.plotly_chart(figure.update_layout(height=SLIDE_FIGURE_HEIGHT))


def render_result(model: DigitClassifier, recognition: Recognition) -> None:
    text_column, image_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        show_formula(
            [
                Term(
                    str(recognition.digit),
                    explanations.OUTPUT_COLOR,
                    f"chiffre reconnu · confiance {recognition.confidence:.0%}",
                )
            ],
            size_rem=RESULT_FORMULA_REM,
        )
    image_column.image(
        digit_image(recognition.pixels), caption="L'image reçue", width=SLIDE_IMAGE_WIDTH
    )
