"""Diaporama de l'accueil : une diapositive à l'écran, boutons précédent et suivant."""

from collections.abc import Callable

import streamlit as st
from network.model import DigitClassifier
from pipeline.recognition import Recognition

from dashboard import reasoning
from dashboard.constants import SLIDE_KEY
from dashboard.live import render_live_network

type SlideRenderer = Callable[[DigitClassifier, Recognition], None]

# La diapositive 0 (choisir un chiffre) est rendue par l'accueil ; les suivantes l'expliquent.
SLIDES: list[tuple[str, SlideRenderer]] = [
    ("Étape 1 · Ce que voit le réseau", reasoning.render_input_step),
    ("Étape 2 · Couche 1 : les neurones repèrent des motifs", reasoning.render_shapes_step),
    ("Étape 2 · Couche 1 : le calcul d'un neurone", reasoning.render_neuron_step),
    ("Étape 3 · Couche 2 : les neurones votent", reasoning.render_vote_step),
    ("Étape 3 · Couche 2 : les pixels qui comptent", reasoning.render_evidence_step),
    ("Étape 4 · La softmax change les scores en pourcentages", reasoning.render_softmax_step),
    ("Résultat", reasoning.render_result),
    ("Conclusion · Le réseau en direct", render_live_network),
]
FIRST_TITLE = "Choisir un chiffre"
SLIDE_COUNT = len(SLIDES) + 1


def current_slide() -> int:
    slide: int = st.session_state.get(SLIDE_KEY, 0)
    return slide


def go_to(slide: int) -> None:
    st.session_state[SLIDE_KEY] = min(max(slide, 0), SLIDE_COUNT - 1)


def slide_title(slide: int) -> str:
    return FIRST_TITLE if slide == 0 else SLIDES[slide - 1][0]


def render_navigation(can_go_on: bool) -> int:
    """Barre du haut : ◀, le titre de la diapositive, ▶ ; renvoie la diapositive affichée."""
    slide = current_slide()
    previous_column, title_column, next_column = st.columns([1, 8, 1], vertical_alignment="center")
    previous_column.button(
        "◀", on_click=go_to, args=(slide - 1,), disabled=slide == 0, help="Diapo précédente"
    )
    title_column.subheader(f"{slide_title(slide)}  ·  {slide + 1}/{SLIDE_COUNT}")
    next_column.button(
        "▶",
        on_click=go_to,
        args=(slide + 1,),
        disabled=slide == SLIDE_COUNT - 1 or not can_go_on,
        help="Diapo suivante",
    )
    return slide


def render_slide(slide: int, model: DigitClassifier, recognition: Recognition) -> None:
    SLIDES[slide - 1][1](model, recognition)
