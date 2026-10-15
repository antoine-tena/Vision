"""Page d'accueil : dessiner, choisir ou importer un chiffre, voir le réseau y réagir."""

import streamlit as st
from network.model import DigitClassifier
from pipeline.figures import probabilities_description, probabilities_figure
from pipeline.recognition import Recognition, recognize, recognize_pixels
from pipeline.transforms import digit_image
from streamlit.elements.lib.mutable_tab_container import TabContainer

from dashboard.canvas import draw_digit
from dashboard.constants import (
    DRAWING_TAB,
    EXAMPLES_TAB,
    INPUT_TABS_KEY,
    PREVIEW_WIDTH,
    SITE_DESCRIPTION,
    SITE_TITLE,
    UPLOAD_TAB,
    UPLOAD_TYPES,
)
from dashboard.examples import pick_example
from dashboard.images import InvalidImageError, decode_image
from dashboard.live import render_live_network
from dashboard.loaders import cached_model
from dashboard.reasoning import render_reasoning


def render_home() -> None:
    st.title(SITE_TITLE)
    st.write(SITE_DESCRIPTION)
    tabs = st.tabs([DRAWING_TAB, EXAMPLES_TAB, UPLOAD_TAB], key=INPUT_TABS_KEY, on_change="rerun")
    model = cached_model()
    try:
        recognition = read_recognition(model, *tabs)
    except InvalidImageError as error:
        st.error(str(error))
        return
    if recognition is None:
        st.info("Tracez un chiffre, choisissez un exemple ou importez une image.")
        return
    if recognition.is_blank:
        st.warning("L'image est vide : aucun trait n'a été trouvé.")
        return
    render_live_network(model, recognition)
    render_recognition(recognition)
    render_reasoning(model, recognition)


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


def render_recognition(recognition: Recognition) -> None:
    answer, preview = st.columns([2, 1])
    answer.metric("Chiffre reconnu", recognition.digit, f"confiance {recognition.confidence:.0%}")
    preview.image(
        digit_image(recognition.pixels),
        caption="Ce que voit le réseau (28 x 28)",
        width=PREVIEW_WIDTH,
    )
    st.plotly_chart(probabilities_figure(recognition.probabilities))
    # [FIGURES-ACCESSIBLES] : texte alternatif de la figure.
    st.caption(probabilities_description(recognition.probabilities))
