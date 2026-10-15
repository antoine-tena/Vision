"""Diapositives de section : le nom de la partie ou de l'étape, son plan, et un visuel propre.

Chaque section a son visuel, pour que rien ne se répète d'une diapositive à l'autre.
"""

from collections.abc import Callable

import numpy as np
import streamlit as st
from network.model import DigitClassifier
from network.training import TrainingConfig
from pipeline.activity import network_activity
from pipeline.figures import network_figure, patterns_figure, scores_figure
from pipeline.reasoning import before_and_after, digit_scores, explain
from pipeline.recognition import Recognition
from pipeline.transforms import digit_image

from dashboard import explanations
from dashboard.constants import SECTION_FIGURE_HEIGHT, SLIDE_IMAGE_WIDTH
from dashboard.explanations import HIDDEN_COLOR, OUTPUT_COLOR, PIXEL_COLOR, key_fact
from dashboard.formula import Term, show_formula

type Visual = Callable[[DigitClassifier, Recognition], None]
type LayerReach = Callable[[tuple[int, int, int]], int]
EXAMPLE_NEURONS = 2


def through_input(sizes: tuple[int, int, int]) -> int:
    return sizes[0]


def through_hidden(sizes: tuple[int, int, int]) -> int:
    return sizes[0] + sizes[1]


def through_output(sizes: tuple[int, int, int]) -> int:
    return sum(sizes)


def network_visual(reach: LayerReach) -> Visual:
    """Le réseau allumé jusqu'à la couche `reach`."""

    def render(model: DigitClassifier, recognition: Recognition) -> None:
        activity = network_activity(model, recognition.pixels)
        figure = network_figure(activity, reach(activity.layer_sizes))
        st.plotly_chart(
            figure.update_layout(height=SECTION_FIGURE_HEIGHT), key=f"reseau-{reach.__name__}"
        )

    return render


def project_visual(model: DigitClassifier, recognition: Recognition) -> None:
    parameters = f"{model.parameter_count:,}".replace(",", " ")
    show_formula(
        [
            Term("image", PIXEL_COLOR, "784 pixels"),
            "&rarr;",
            Term("réseau", HIDDEN_COLOR, f"{parameters} paramètres"),
            "&rarr;",
            Term("chiffre", OUTPUT_COLOR, "un parmi 10"),
        ],
        size_rem=2.2,
    )


def learning_visual(model: DigitClassifier, recognition: Recognition) -> None:
    """Deux neurones : leurs poids tirés au hasard, puis appris."""
    neurons = explain(model, recognition.pixels, recognition.digit).neuron_ids[:EXAMPLE_NEURONS]
    patterns = before_and_after(model, neurons, TrainingConfig().seed)
    titles = [f"n°{neuron}<br>{moment}" for neuron in neurons for moment in ("au départ", "appris")]
    figure = patterns_figure(patterns, titles, "Un neurone, avant et après l'entraînement")
    st.plotly_chart(figure.update_layout(height=SECTION_FIGURE_HEIGHT))


def image_visual(model: DigitClassifier, recognition: Recognition) -> None:
    st.image(
        digit_image(recognition.pixels), caption="Un chiffre à reconnaître", width=SLIDE_IMAGE_WIDTH
    )


def scores_visual(model: DigitClassifier, recognition: Recognition) -> None:
    scores = digit_scores(model, np.asarray(recognition.pixels))
    figure = scores_figure(scores).update_layout(height=SECTION_FIGURE_HEIGHT)
    st.plotly_chart(figure)


def section(name: str, visual: Visual, agenda: list[str], noun: str = "étape") -> Visual:
    """Diapositive de section : `name` en gros, le plan à gauche, le visuel à droite."""

    def render(model: DigitClassifier, recognition: Recognition) -> None:
        st.markdown(f"<h1 style='text-align:center'>{name}</h1>", unsafe_allow_html=True)
        agenda_column, visual_column = st.columns([2, 3], vertical_alignment="center")
        with agenda_column:
            key_fact(f"Dans cette {noun}")
            explanations.explain_list("\n".join(f"- **{title}**" for title in agenda))
        with visual_column:
            visual(model, recognition)

    return render
