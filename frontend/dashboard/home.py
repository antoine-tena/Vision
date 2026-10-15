"""La présentation : le diaporama, du projet jusqu'au chiffre reconnu."""

import streamlit as st
from pipeline.recognition import Recognition

from dashboard.choice import example_recognition
from dashboard.constants import RECOGNITION_KEY, SLIDE_PAGE_STYLE
from dashboard.loaders import cached_model
from dashboard.slides import CHOICE_INDEX, current_slide, go_to, render_navigation, render_slide

AUTHORS = "Antoine Tena et Nathan Barrachin"


def render_home() -> None:
    # Diaporama : marges du haut et du bas réduites pour qu'une diapositive tienne à l'écran.
    st.markdown(SLIDE_PAGE_STYLE, unsafe_allow_html=True)
    model = cached_model()
    chosen: Recognition | None = st.session_state.get(RECOGNITION_KEY)
    if chosen is None and current_slide() > CHOICE_INDEX:
        go_to(CHOICE_INDEX)
    slide = render_navigation(can_go_on=chosen is not None or current_slide() != CHOICE_INDEX)
    # Avant le choix, les diapositives illustrent avec un exemple d'entraînement.
    recognition = chosen or example_recognition(model)
    if slide == 0:
        st.markdown(f"**{AUTHORS}**")
    if recognition is not None:
        render_slide(slide, model, recognition)
