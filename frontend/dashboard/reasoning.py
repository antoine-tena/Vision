"""Section « Raisonnement interne » de l'accueil : comment le réseau est arrivé à sa réponse."""

import streamlit as st
from network.model import DIGIT_COUNT, PIXEL_COUNT, DigitClassifier
from pipeline.figures import (
    evidence_description,
    evidence_figure,
    neurons_description,
    neurons_figure,
)
from pipeline.reasoning import explain
from pipeline.recognition import Recognition

from dashboard import explanations


def render_reasoning(model: DigitClassifier, recognition: Recognition) -> None:
    st.subheader("Raisonnement interne")
    # Sans clé : quand le chiffre reconnu change, le choix revient sur lui.
    digit = st.selectbox(
        "Expliquer le score du chiffre",
        range(DIGIT_COUNT),
        index=recognition.digit,
        help="Par défaut le chiffre reconnu ; choisir un autre chiffre montre ce qui l'a écarté.",
    )
    reasoning = explain(model, recognition.pixels, digit)
    st.markdown(
        f"1. **Entrée** : l'image ramenée à 28 x 28 devient {PIXEL_COUNT} nombres entre 0 et 1.\n"
        f"2. **Couche cachée** : {reasoning.active_count} neurones sur {reasoning.hidden_size} "
        "s'allument ; chacun guette un motif de traits.\n"
        f"3. **Vote** : chaque neurone allumé vote pour ou contre le {digit} ; le total donne "
        f"son score, **{reasoning.score:+.2f}**.\n"
        "4. **Décision** : les dix scores passent par un softmax, qui en fait les probabilités "
        "ci-dessus."
    )
    render_active_shapes(reasoning.active_shapes)
    evidence_column, neurons_column = st.columns(2)
    with evidence_column:
        st.plotly_chart(evidence_figure(reasoning.evidence, digit))
        st.caption(evidence_description(reasoning.evidence, digit))
        explanations.explain(explanations.WEIGHT)
    with neurons_column:
        st.plotly_chart(neurons_figure(reasoning))
        st.caption(neurons_description(reasoning))
        explanations.explain(explanations.NEURON_VOTES)


def render_active_shapes(shapes: list[tuple[str, int]]) -> None:
    """Les motifs guettés par les neurones allumés, regroupés et comptés."""
    lines = "\n".join(f"- **{count}** x {shape}" for shape, count in shapes)
    st.markdown(f"**Ce que les neurones allumés ont reconnu dans l'image :**\n{lines}")
    explanations.explain(explanations.SHAPES)
