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
    LayerReach,
    section,
    through_hidden,
    through_input,
    through_output,
)

type SlideRenderer = Callable[[DigitClassifier, Recognition], None]
type Group = tuple[str, LayerReach, list[tuple[str, SlideRenderer]]]

PROJECT: Group = (
    "Le projet",
    through_output,
    [
        ("Ce que c'est", project.render_what_slide),
        ("L'architecture", project.render_architecture_slide),
        ("Le lien avec les LLM", project.render_llm_slide),
    ],
)
LEARNING: Group = (
    "Comment il apprend",
    through_output,
    [
        ("Les 4 temps de l'apprentissage", learning.render_loop_slide),
        ("La perte", learning.render_loss_slide),
        ("La descente de gradient", learning.render_descent_slide),
        ("Les données", training.render_split_slide),
        ("Les courbes d'apprentissage", training.render_curves_slide),
        ("Le test", training.render_test_slide),
    ],
)
# Les étapes du raisonnement, sur le chiffre choisi par l'utilisateur.
STEPS: list[Group] = [
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
CHOICE_TITLE = "Choisir un chiffre"
RECOGNITION_AGENDA = [
    CHOICE_TITLE,
    *(f"Étape {number} · {name}" for number, (name, _, _) in enumerate(STEPS, start=1)),
    "Résultat",
]


def group_slides(
    label: str, number: int, group: Group, sub_label: str = ""
) -> list[tuple[str, SlideRenderer]]:
    """La diapositive de section (avec le plan), puis les sous-parties numérotées n.1, n.2…"""
    name, reach, parts = group
    titles = [f"{number}.{index} · {title}" for index, (title, _) in enumerate(parts, 1)]
    return [
        (f"{label} {number}", section(name, reach, titles)),
        *(
            (f"{sub_label}{title}", render)
            for title, (_, render) in zip(titles, parts, strict=True)
        ),
    ]


SLIDES: list[tuple[str, SlideRenderer]] = [
    *group_slides("Partie", 1, PROJECT),
    *group_slides("Partie", 2, LEARNING),
    ("Partie 3", section("Reconnaître un chiffre", through_output, RECOGNITION_AGENDA)),
    (CHOICE_TITLE, choice.render_choice_slide),
    *(
        slide
        for number, step in enumerate(STEPS, start=1)
        for slide in group_slides("Étape", number, step, sub_label="Étape ")
    ),
    ("Résultat", reasoning.render_result),
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
