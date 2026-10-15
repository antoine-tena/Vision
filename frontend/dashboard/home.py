"""Page d'accueil : dessiner ou importer un chiffre, voir ce que le réseau en pense."""

import streamlit as st
from network.model import DEFAULT_MODEL_PATH, DigitClassifier, load_model
from PIL import Image
from pipeline.figures import probabilities_description, probabilities_figure
from pipeline.recognition import Recognition, recognize
from pipeline.transforms import digit_image
from streamlit.delta_generator import DeltaGenerator

from dashboard.canvas import draw_digit
from dashboard.constants import PREVIEW_WIDTH, SITE_DESCRIPTION, SITE_TITLE, UPLOAD_TYPES
from dashboard.images import InvalidImageError, decode_image


# [CACHE] : le modèle se charge une fois, pas à chaque trait.
@st.cache_resource
def cached_model() -> DigitClassifier:
    """Le réseau entraîné, chargé depuis `vision.npz`."""
    return load_model(DEFAULT_MODEL_PATH)


def render_home() -> None:
    """Construit la page : titre, onglets Dessiner et Importer, puis la réponse du réseau.

    Une image refusée affiche une erreur ; sans image, un message invite à dessiner.
    """
    st.set_page_config(page_title=SITE_TITLE)
    st.title(SITE_TITLE)
    st.write(SITE_DESCRIPTION)
    drawing_tab, upload_tab = st.tabs(["Dessiner", "Importer une image"])
    try:
        image = read_image(drawing_tab, upload_tab)
    except InvalidImageError as error:
        st.error(str(error))
        return
    if image is None:
        st.info("Tracez un chiffre dans la zone de dessin, ou importez une image.")
        return
    render_recognition(recognize(cached_model(), image))


def read_image(drawing_tab: DeltaGenerator, upload_tab: DeltaGenerator) -> Image.Image | None:
    """Le fichier importé s'il y en a un, sinon le dessin ; InvalidImageError s'il est refusé."""
    with drawing_tab:
        drawn = draw_digit()
    with upload_tab:
        uploaded = st.file_uploader(
            "Image d'un chiffre (manuscrit ou imprimé)", type=list(UPLOAD_TYPES)
        )
    return decode_image(uploaded.getvalue()) if uploaded is not None else drawn


def render_recognition(recognition: Recognition) -> None:
    """Affiche le chiffre reconnu et sa confiance, l'image 28x28 vue par le réseau et le
    graphique des probabilités, avec son texte alternatif."""
    if recognition.is_blank:
        st.warning("L'image est vide : aucun trait n'a été trouvé.")
        return
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
