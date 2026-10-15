"""La présentation en diaporama : le projet, son apprentissage, puis un chiffre reconnu étape
par étape. Une diapositive à l'écran, boutons précédent et suivant (ou le clavier)."""

from collections.abc import Callable

import streamlit as st
from network.model import DigitClassifier
from pipeline.recognition import Recognition

from dashboard import choice, learning, project, reasoning, training
from dashboard.constants import EXTRA_NEXT_SHORTCUTS, SHORTCUTS_KEY, SLIDE_KEY
from dashboard.live import render_live_network
from dashboard.sections import (
    Visual,
    image_visual,
    learning_visual,
    network_visual,
    project_visual,
    scores_visual,
    section,
    through_hidden,
    through_input,
    through_output,
)

type SlideRenderer = Callable[[DigitClassifier, Recognition], None]
type Group = tuple[str, Visual, list[tuple[str, SlideRenderer]]]

PROJECT: Group = (
    "Le projet",
    project_visual,
    [
        ("Ce que c'est", project.render_what_slide),
        ("L'architecture", project.render_architecture_slide),
        ("Le lien avec les LLM", project.render_llm_slide),
    ],
)
LEARNING: Group = (
    "Comment il apprend",
    learning_visual,
    [
        ("Les 4 temps de l'apprentissage", learning.render_loop_slide),
        ("La perte", learning.render_loss_slide),
        ("La descente de gradient", learning.render_descent_slide),
        ("Les données", training.render_split_slide),
        ("Les courbes d'apprentissage", training.render_curves_slide),
        ("Le test", training.render_test_slide),
    ],
)
# Partie 3 : les étapes du raisonnement, sur le chiffre choisi (3.2 à 3.5).
STEPS: list[Group] = [
    (
        "L'entrée : ce que reçoit le réseau",
        network_visual(through_input),
        [("Ce que voit le réseau", reasoning.render_input_step)],
    ),
    (
        "Couche 1 : repérer des motifs",
        network_visual(through_hidden),
        [
            ("Les neurones allumés et leurs motifs", reasoning.render_shapes_step),
            ("Le calcul d'un neurone", reasoning.render_neuron_step),
            ("Qui a choisi les motifs ? Personne", reasoning.render_emergence_step),
        ],
    ),
    (
        "Couche 2 : voter pour un chiffre",
        network_visual(through_output),
        [
            ("Les deux niveaux de poids", reasoning.render_two_levels_step),
            ("Les votes des neurones", reasoning.render_vote_step),
            ("Les pixels qui ont compté", reasoning.render_evidence_step),
        ],
    ),
    (
        "La softmax : des scores aux pourcentages",
        scores_visual,
        [("Le calcul de la softmax", reasoning.render_softmax_step)],
    ),
]
CHOICE_TITLE = "3.1 · Choisir un chiffre"
FIRST_STEP = 2
RESULT_TITLE = f"3.{FIRST_STEP + len(STEPS)} · Résultat"
RECOGNITION_AGENDA = [
    CHOICE_TITLE,
    *(f"3.{number} · {name}" for number, (name, _, _) in enumerate(STEPS, start=FIRST_STEP)),
    RESULT_TITLE,
]


def group_slides(
    label: str, prefix: str, group: Group, noun: str
) -> list[tuple[str, SlideRenderer]]:
    """La diapositive de section (titre `label`, avec le plan), puis les sous-parties
    numérotées `prefix`.1, `prefix`.2…"""
    name, visual, parts = group
    titles = [f"{prefix}.{index} · {title}" for index, (title, _) in enumerate(parts, 1)]
    return [
        (label, section(name, visual, titles, noun)),
        *((title, render) for title, (_, render) in zip(titles, parts, strict=True)),
    ]


SLIDES: list[tuple[str, SlideRenderer]] = [
    *group_slides("Partie 1", "1", PROJECT, "partie"),
    *group_slides("Partie 2", "2", LEARNING, "partie"),
    ("Partie 3", section("Reconnaître un chiffre", image_visual, RECOGNITION_AGENDA, "partie")),
    (CHOICE_TITLE, choice.render_choice_slide),
    *(
        slide
        for number, step in enumerate(STEPS, start=FIRST_STEP)
        for slide in group_slides(f"3.{number}", f"3.{number}", step, "étape")
    ),
    (RESULT_TITLE, reasoning.render_result),
    ("Conclusion · Le réseau en direct", render_live_network),
]
CHOICE_INDEX = [title for title, _ in SLIDES].index(CHOICE_TITLE)
SLIDE_COUNT = len(SLIDES)


def current_slide() -> int:
    slide: int = st.session_state.get(SLIDE_KEY, 0)
    return slide


def go_to(slide: int) -> None:
    st.session_state[SLIDE_KEY] = min(max(slide, 0), SLIDE_COUNT - 1)


def slide_title(slide: int) -> str:
    return SLIDES[slide][0]


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
    SLIDES[slide][1](model, recognition)
