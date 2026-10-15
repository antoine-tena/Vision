"""Diaporama de l'accueil : une diapositive à l'écran, boutons précédent et suivant."""

from collections.abc import Callable

import streamlit as st
from network.model import DigitClassifier
from pipeline.recognition import Recognition

from dashboard import reasoning
from dashboard.constants import EXTRA_NEXT_SHORTCUTS, SHORTCUTS_KEY, SLIDE_KEY
from dashboard.live import render_live_network
from dashboard.sections import (
    LayerReach,
    section,
    through_hidden,
    through_input,
    through_output,
)

type SlideRenderer = Callable[[DigitClassifier, Recognition], None]

# Chaque étape : son nom, la couche allumée sur sa diapositive de section, ses sous-étapes.
STEPS: list[tuple[str, LayerReach, list[tuple[str, SlideRenderer]]]] = [
    (
        "L'entrée : l'image",
        through_input,
        [("Ce que voit le réseau", reasoning.render_input_step)],
    ),
    (
        "Couche 1 : repérer des motifs",
        through_hidden,
        [
            ("Les neurones repèrent des motifs", reasoning.render_shapes_step),
            ("Le calcul d'un neurone", reasoning.render_neuron_step),
            ("Qui a choisi les motifs ? Personne", reasoning.render_emergence_step),
        ],
    ),
    (
        "Couche 2 : voter pour un chiffre",
        through_output,
        [
            ("Les neurones votent", reasoning.render_vote_step),
            ("Les pixels qui comptent", reasoning.render_evidence_step),
        ],
    ),
    (
        "La softmax : des pourcentages",
        through_output,
        [("Des scores aux pourcentages", reasoning.render_softmax_step)],
    ),
]


def build_slides() -> list[tuple[str, SlideRenderer]]:
    """Pour chaque étape : sa diapositive de section (avec le plan), puis ses sous-étapes."""
    slides: list[tuple[str, SlideRenderer]] = []
    for number, (name, reach, parts) in enumerate(STEPS, start=1):
        titles = [f"{number}.{index} · {title}" for index, (title, _) in enumerate(parts, 1)]
        slides.append((f"Étape {number}", section(name, reach, titles)))
        slides += [
            (f"Étape {title}", render) for title, (_, render) in zip(titles, parts, strict=True)
        ]
    return slides


# La diapositive 0 (choisir un chiffre) est rendue par l'accueil ; les suivantes l'expliquent.
SLIDES = [
    *build_slides(),
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
        "◀ Préc.",
        on_click=go_to,
        args=(slide - 1,),
        disabled=slide == 0,
        help="Diapo précédente (←)",
        shortcut="Left",
        width="stretch",
    )
    title_column.subheader(slide_title(slide))
    next_column.button(
        "Suiv. ▶",
        on_click=go_to,
        args=(slide + 1,),
        disabled=slide == SLIDE_COUNT - 1 or not can_go_on,
        help="Diapo suivante (→, Entrée ou Espace)",
        shortcut="Right",
        type="primary",
        width="stretch",
    )
    render_extra_shortcuts(slide, can_go_on)
    return slide


def render_extra_shortcuts(slide: int, can_go_on: bool) -> None:
    """Entrée et Espace avancent aussi : un bouton par touche, masqué (voir SLIDE_PAGE_STYLE)."""
    is_disabled = slide == SLIDE_COUNT - 1 or not can_go_on
    with st.container(key=SHORTCUTS_KEY):
        for shortcut in EXTRA_NEXT_SHORTCUTS:
            st.button(
                f"Suivante ({shortcut})",
                key=f"suivante-{shortcut}",
                on_click=go_to,
                args=(slide + 1,),
                disabled=is_disabled,
                shortcut=shortcut,
            )


def render_slide(slide: int, model: DigitClassifier, recognition: Recognition) -> None:
    SLIDES[slide - 1][1](model, recognition)
