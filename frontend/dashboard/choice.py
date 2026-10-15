"""Diapositive « Choisir un chiffre » : dessiner, prendre un exemple ou importer une image."""

import streamlit as st
from network.model import DigitClassifier
from pipeline.recognition import Recognition, recognize, recognize_pixels
from pipeline.sources import DatasetFormatError
from pipeline.transforms import digit_image
from streamlit.elements.lib.mutable_tab_container import TabContainer

from dashboard.canvas import draw_digit
from dashboard.constants import (
    DRAWING_TAB,
    EXAMPLES_TAB,
    INPUT_TABS_KEY,
    MISSING_SAMPLE,
    PREVIEW_WIDTH,
    RECOGNITION_KEY,
    UPLOAD_TAB,
    UPLOAD_TYPES,
)
from dashboard.examples import pick_example
from dashboard.images import InvalidImageError, decode_image
from dashboard.loaders import cached_training_sample


def render_choice_slide(model: DigitClassifier, example: Recognition) -> None:
    """Dessiner, choisir ou importer un chiffre ; le garde pour les diapositives suivantes.
    `example` (l'exemple affiché tant que rien n'est choisi) n'est pas utilisé ici."""
    tabs = st.tabs([DRAWING_TAB, EXAMPLES_TAB, UPLOAD_TAB], key=INPUT_TABS_KEY, on_change="rerun")
    try:
        chosen = read_recognition(model, *tabs)
    except InvalidImageError as error:
        st.error(str(error))
        return
    if chosen is None:
        st.info("Tracez un chiffre, choisissez un exemple ou importez une image, puis ▶.")
        return
    if chosen.is_blank:
        st.warning("L'image est vide : aucun trait n'a été trouvé.")
        return
    remember(chosen)
    st.image(digit_image(chosen.pixels), caption="Ce que voit le réseau", width=PREVIEW_WIDTH)
    st.success(f"Chiffre prêt : {chosen.digit}. Suivant (→) pour suivre le raisonnement.")


def remember(recognition: Recognition) -> None:
    """Garde le chiffre pour les diapositives ; au premier, relance pour activer ▶."""
    had_none = st.session_state.get(RECOGNITION_KEY) is None
    st.session_state[RECOGNITION_KEY] = recognition
    if had_none:
        st.rerun()


def read_recognition(
    model: DigitClassifier,
    drawing_tab: TabContainer,
    examples_tab: TabContainer,
    upload_tab: TabContainer,
) -> Recognition | None:
    """L'image de l'onglet ouvert, reconnue ; InvalidImageError si un fichier est refusé."""
    if examples_tab.open:
        with examples_tab:
            pixels = pick_example()
        return None if pixels is None else recognize_pixels(model, pixels)
    if upload_tab.open:
        with upload_tab:
            uploaded = st.file_uploader(
                "Image d'un chiffre (manuscrit ou imprimé)", type=list(UPLOAD_TYPES)
            )
        return None if uploaded is None else recognize(model, decode_image(uploaded.getvalue()))
    with drawing_tab:
        drawn = draw_digit()
    return None if drawn is None else recognize(model, drawn)


def example_recognition(model: DigitClassifier) -> Recognition | None:
    """Le premier exemple d'entraînement, reconnu : il illustre les diapositives d'avant le
    choix. None (et un message) si l'échantillon manque."""
    try:
        sample = cached_training_sample()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_SAMPLE.format(error=error))
        return None
    first_source = next(iter(sample.values()))
    return recognize_pixels(model, first_source.images[:1])
