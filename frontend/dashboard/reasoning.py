"""Le raisonnement du réseau, étape par étape, de l'image reçue jusqu'au chiffre reconnu."""

import streamlit as st
from network.model import DIGIT_COUNT, PIXEL_COUNT, DigitClassifier
from pipeline.figures import evidence_figure, neurons_figure, probabilities_figure
from pipeline.reasoning import Reasoning, evidence_totals, explain
from pipeline.recognition import Recognition
from pipeline.transforms import digit_image

from dashboard import explanations
from dashboard.constants import PREVIEW_WIDTH, TOP_SHAPES
from dashboard.explanations import key_fact
from dashboard.formula import show_formula


def render_steps(model: DigitClassifier, recognition: Recognition) -> None:
    render_input_step(recognition)
    reasoning = render_hidden_step(model, recognition)
    render_vote_step(model, recognition, reasoning)
    render_softmax_step(recognition)
    render_result(recognition)


def render_input_step(recognition: Recognition) -> None:
    st.header("Étape 1 · Ce que voit le réseau")
    image_column, text_column = st.columns([1, 2])
    image_column.image(
        digit_image(recognition.pixels), caption="L'image en 28 x 28", width=PREVIEW_WIDTH
    )
    with text_column:
        key_fact(f"28 x 28 = {PIXEL_COUNT} pixels → {PIXEL_COUNT} nombres entre 0 et 1")
        explanations.explain(explanations.INPUT)


def render_hidden_step(model: DigitClassifier, recognition: Recognition) -> Reasoning:
    st.header("Étape 2 · Couche 1 : les neurones repèrent des motifs")
    reasoning = explain(model, recognition.pixels, recognition.digit)
    key_fact(f"{reasoning.active_count} neurones allumés sur {reasoning.hidden_size}")
    st.markdown(
        " · ".join(f"**{count}** {shape}" for shape, count in reasoning.active_shapes[:TOP_SHAPES])
    )
    explanations.explain(explanations.SHAPES)
    key_fact("Chaque neurone calcule une note, puis s'allume si elle est positive")
    show_formula(explanations.NEURON_FORMULA)
    show_formula(explanations.RELU_FORMULA)
    explanations.explain(explanations.WEIGHT_SIGNS)
    return reasoning


def render_vote_step(
    model: DigitClassifier, recognition: Recognition, reasoning: Reasoning
) -> None:
    st.header("Étape 3 · Couche 2 : les neurones votent pour chaque chiffre")
    # Sans clé : quand le chiffre reconnu change, le choix revient sur lui.
    digit = st.selectbox(
        "Voir les votes pour le chiffre",
        range(DIGIT_COUNT),
        index=recognition.digit,
        help="Par défaut le chiffre reconnu ; choisir un autre chiffre montre ce qui l'a écarté.",
    )
    if digit != reasoning.digit:
        reasoning = explain(model, recognition.pixels, digit)
    key_fact(f"Score du {digit} = somme des votes = {reasoning.score:+.2f}")
    show_formula(explanations.VOTE_FORMULA)
    st.plotly_chart(neurons_figure(reasoning))
    explanations.explain(explanations.NEURON_VOTES)
    support, against = evidence_totals(reasoning.evidence)
    key_fact(f"Pixels pour le {digit} : +{support:.1f} · contre : -{against:.1f}")
    st.plotly_chart(evidence_figure(reasoning.evidence, digit))
    explanations.explain(explanations.EVIDENCE)


def render_softmax_step(recognition: Recognition) -> None:
    st.header("Étape 4 · La softmax change les scores en pourcentages")
    key_fact(f"{recognition.digit} : {recognition.confidence:.0%} des chances")
    show_formula(explanations.SOFTMAX_FORMULA)
    st.plotly_chart(probabilities_figure(recognition.probabilities))
    explanations.explain(explanations.PROBABILITIES)


def render_result(recognition: Recognition) -> None:
    st.header("Résultat")
    st.metric("Chiffre reconnu", recognition.digit, f"confiance {recognition.confidence:.0%}")
