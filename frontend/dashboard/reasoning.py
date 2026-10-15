"""Le raisonnement du réseau, étape par étape, de l'image reçue jusqu'au chiffre reconnu."""

import streamlit as st
from network.model import DIGIT_COUNT, DigitClassifier
from pipeline.figures import (
    evidence_description,
    evidence_figure,
    neurons_description,
    neurons_figure,
    probabilities_description,
    probabilities_figure,
)
from pipeline.reasoning import Reasoning, explain
from pipeline.recognition import Recognition
from pipeline.transforms import digit_image

from dashboard import explanations
from dashboard.constants import PREVIEW_WIDTH, TOP_SHAPES


def render_steps(model: DigitClassifier, recognition: Recognition) -> None:
    render_input_step(recognition)
    reasoning = render_hidden_step(model, recognition)
    render_vote_step(model, recognition, reasoning)
    render_softmax_step(recognition)
    render_result(recognition)


def render_input_step(recognition: Recognition) -> None:
    st.header("Étape 1 · Ce que voit le réseau")
    st.image(digit_image(recognition.pixels), caption="L'image en 28 x 28", width=PREVIEW_WIDTH)
    explanations.explain(explanations.INPUT)


def render_hidden_step(model: DigitClassifier, recognition: Recognition) -> Reasoning:
    st.header("Étape 2 · Couche 1 : les neurones repèrent des motifs")
    reasoning = explain(model, recognition.pixels, recognition.digit)
    st.markdown(f"**{reasoning.active_count} / {reasoning.hidden_size} allumés** · motifs :")
    st.markdown(
        "\n".join(
            f"- **{count}** x {shape}" for shape, count in reasoning.active_shapes[:TOP_SHAPES]
        )
    )
    explanations.explain(explanations.HIDDEN_LAYER)
    explanations.explain(explanations.SHAPES)
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
    st.markdown(f"Score du {digit} (somme des votes) : **{reasoning.score:+.2f}**")
    st.plotly_chart(neurons_figure(reasoning))
    explanations.describe(neurons_description(reasoning))
    explanations.explain(explanations.NEURON_VOTES)
    st.plotly_chart(evidence_figure(reasoning.evidence, digit))
    explanations.describe(evidence_description(reasoning.evidence, digit))
    explanations.explain(explanations.WEIGHT)


def render_softmax_step(recognition: Recognition) -> None:
    st.header("Étape 4 · La softmax change les scores en pourcentages")
    st.plotly_chart(probabilities_figure(recognition.probabilities))
    explanations.describe(probabilities_description(recognition.probabilities))
    explanations.explain(explanations.PROBABILITIES)


def render_result(recognition: Recognition) -> None:
    st.header("Résultat")
    st.metric("Chiffre reconnu", recognition.digit, f"confiance {recognition.confidence:.0%}")
