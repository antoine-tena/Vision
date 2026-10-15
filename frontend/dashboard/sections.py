"""Diapositives de section : le nom de l'étape et le réseau allumé jusqu'à elle, sans texte."""

from collections.abc import Callable

import streamlit as st
from network.model import DigitClassifier
from pipeline.activity import network_activity
from pipeline.figures import network_figure
from pipeline.recognition import Recognition

from dashboard.constants import SECTION_FIGURE_HEIGHT
from dashboard.explanations import explain, key_fact

type LayerReach = Callable[[tuple[int, int, int]], int]


def through_input(sizes: tuple[int, int, int]) -> int:
    return sizes[0]


def through_hidden(sizes: tuple[int, int, int]) -> int:
    return sizes[0] + sizes[1]


def through_output(sizes: tuple[int, int, int]) -> int:
    return sum(sizes)


def section(
    name: str, reach: LayerReach, agenda: list[str]
) -> Callable[[DigitClassifier, Recognition], None]:
    """Diapositive de section : `name` en gros, le plan de l'étape (ses sous-étapes) à gauche,
    le réseau allumé jusqu'à la couche `reach` à droite."""

    def render(model: DigitClassifier, recognition: Recognition) -> None:
        st.markdown(f"<h1 style='text-align:center'>{name}</h1>", unsafe_allow_html=True)
        agenda_column, figure_column = st.columns([2, 3], vertical_alignment="center")
        with agenda_column:
            key_fact("Dans cette étape")
            explain("\n".join(f"- **{title}**" for title in agenda))
        activity = network_activity(model, recognition.pixels)
        figure = network_figure(activity, reach(activity.layer_sizes))
        figure_column.plotly_chart(
            figure.update_layout(height=SECTION_FIGURE_HEIGHT), key=f"section-{name}"
        )

    return render
