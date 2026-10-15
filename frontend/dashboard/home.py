"""Page d'accueil : dessiner, choisir ou importer un chiffre, voir le réseau y réagir."""

import streamlit as st
from network.model import DigitClassifier
from pipeline.recognition import Recognition, recognize, recognize_pixels
from pipeline.transforms import digit_image
from streamlit.elements.lib.mutable_tab_container import TabContainer

from dashboard.canvas import draw_digit
from dashboard.constants import (
    DRAWING_TAB,
    EXAMPLES_TAB,
    INPUT_TABS_KEY,
    PREVIEW_WIDTH,
    RECOGNITION_KEY,
    SITE_DESCRIPTION,
    SLIDE_PAGE_STYLE,
    UPLOAD_TAB,
    UPLOAD_TYPES,
)
from dashboard.examples import pick_example
from dashboard.images import InvalidImageError, decode_image
from dashboard.loaders import cached_model
from dashboard.slides import go_to, render_navigation, render_slide


def render_home() -> None:
    # Diaporama : marges du haut et du bas réduites pour qu'une diapositive tienne à l'écran.
    st.markdown(SLIDE_PAGE_STYLE, unsafe_allow_html=True)
    model = cached_model()
    recognition: Recognition | None = st.session_state.get(RECOGNITION_KEY)
    slide = render_navigation(can_go_on=recognition is not None)
    if slide == 0:
        render_input(model)
    elif recognition is None:
        go_to(0)
        st.rerun()
    else:
        render_slide(slide, model, recognition)


def render_input(model: DigitClassifier) -> None:
    """Première diapositive : dessiner, choisir ou importer ; garde le chiffre reconnu."""
    st.caption(SITE_DESCRIPTION)
    tabs = st.tabs([DRAWING_TAB, EXAMPLES_TAB, UPLOAD_TAB], key=INPUT_TABS_KEY, on_change="rerun")
    try:
        recognition = read_recognition(model, *tabs)
    except InvalidImageError as error:
        st.error(str(error))
        return
    if recognition is None:
        st.info("Tracez un chiffre, choisissez un exemple ou importez une image, puis ▶.")
        return
    if recognition.is_blank:
        st.warning("L'image est vide : aucun trait n'a été trouvé.")
        return
    remember(recognition)
    st.image(digit_image(recognition.pixels), caption="Ce que voit le réseau", width=PREVIEW_WIDTH)
    st.success(f"Chiffre prêt : {recognition.digit}. Cliquez sur ▶ pour suivre le raisonnement.")


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
