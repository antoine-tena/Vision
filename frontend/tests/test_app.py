import base64
import io

import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from dashboard.images import (
    DATA_URL_PREFIX,
    MAX_IMAGE_BYTES,
    InvalidImageError,
    decode_data_url,
    decode_image,
)

# Relatif à ce fichier de test, comme l'attend AppTest.from_file.
APP_SCRIPT = "../app.py"


def test_app_opens_on_the_first_part_of_the_presentation() -> None:
    app = AppTest.from_file(APP_SCRIPT).run(timeout=20)

    assert not app.exception
    assert app.subheader[0].value == "Partie 1"
    assert any("Le projet" in text.value for text in app.markdown)


def render_the_home_page() -> None:
    # Onglet d'import : la zone de dessin n'est enregistrée que dans l'app complète.
    import streamlit as st

    from dashboard.constants import INPUT_TABS_KEY, SLIDE_KEY, UPLOAD_TAB
    from dashboard.home import render_home
    from dashboard.slides import CHOICE_INDEX

    st.session_state.setdefault(INPUT_TABS_KEY, UPLOAD_TAB)
    st.session_state.setdefault(SLIDE_KEY, CHOICE_INDEX)
    render_home()


def test_home_invites_to_draw_or_upload() -> None:
    app = AppTest.from_function(render_the_home_page).run()

    assert not app.exception
    assert app.subheader[0].value == "Choisir un chiffre"
    assert "Tracez un chiffre" in app.info[0].value
    assert app.button[1].disabled


def test_canvas_data_url_is_decoded() -> None:
    image = Image.new("L", (10, 10), 0)
    ImageDraw.Draw(image).line((2, 2, 7, 7), fill=255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = DATA_URL_PREFIX + base64.b64encode(buffer.getvalue()).decode()

    assert decode_data_url(data_url).size == (10, 10)


def render_a_recognized_digit() -> None:
    # Script lancé à part par AppTest : ses imports vivent dans la fonction.
    from network.model import DEFAULT_MODEL_PATH, load_model
    from PIL import Image, ImageDraw
    from pipeline.recognition import recognize

    from dashboard.reasoning import render_result, render_softmax_step

    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).line((50, 15, 50, 85), fill=0, width=10)
    model = load_model(DEFAULT_MODEL_PATH)
    recognition = recognize(model, image)
    render_softmax_step(model, recognition)
    render_result(model, recognition)


def test_recognition_shows_the_digit_and_its_probabilities() -> None:
    app = AppTest.from_function(render_a_recognized_digit).run()

    assert not app.exception
    assert any(text.value.endswith("des chances") for text in app.markdown)


@pytest.mark.parametrize(
    "payload", [b"pas une image", b"\x89PNG" + bytes(MAX_IMAGE_BYTES)], ids=["illisible", "lourde"]
)
def test_refused_images_raise_a_readable_error(payload: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode_image(payload)


def test_garbled_drawing_is_refused() -> None:
    with pytest.raises(InvalidImageError):
        decode_data_url(DATA_URL_PREFIX + "%%%")
