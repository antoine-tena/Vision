"""Page d'accueil : dessiner ou importer un chiffre, voir ce que le réseau en pense."""

import numpy as np
import streamlit as st
from network.model import DEFAULT_MODEL_PATH, DigitClassifier, load_model
from PIL import Image
from pipeline.figures import digit_image, probabilities_figure
from pipeline.recognition import Recognition, recognize

from dashboard.canvas import draw_digit
from dashboard.constants import PREVIEW_WIDTH, SITE_DESCRIPTION, SITE_TITLE, UPLOAD_TYPES


# [CACHE] : le modèle se charge une fois, pas à chaque trait.
@st.cache_resource
def cached_model() -> DigitClassifier:
    return load_model(DEFAULT_MODEL_PATH)


def render_home() -> None:
    st.set_page_config(page_title=SITE_TITLE)
    st.title(SITE_TITLE)
    st.write(SITE_DESCRIPTION)
    drawing_tab, upload_tab = st.tabs(["Dessiner", "Importer une image"])
    with drawing_tab:
        drawn = draw_digit()
    with upload_tab:
        uploaded = st.file_uploader(
            "Image d'un chiffre (manuscrit ou imprimé)", type=list(UPLOAD_TYPES)
        )
    image = Image.open(uploaded) if uploaded is not None else drawn
    if image is None:
        st.info("Tracez un chiffre dans la zone de dessin, ou importez une image.")
        return
    render_recognition(recognize(cached_model(), image))


def render_recognition(recognition: Recognition) -> None:
    if recognition.is_blank:
        st.warning("L'image est vide : aucun trait n'a été trouvé.")
        return
    answer, preview = st.columns([2, 1])
    answer.metric("Chiffre reconnu", recognition.digit, f"confiance {recognition.confidence:.0%}")
    preview.image(
        np.clip(digit_image(recognition.pixels), 0.0, 1.0),
        caption="Ce que voit le réseau (28 x 28)",
        width=PREVIEW_WIDTH,
    )
    st.plotly_chart(probabilities_figure(recognition.probabilities))
