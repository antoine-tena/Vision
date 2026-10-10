import base64
import io

import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from dashboard.constants import SITE_TITLE
from dashboard.images import (
    DATA_URL_PREFIX,
    MAX_IMAGE_BYTES,
    InvalidImageError,
    decode_data_url,
    decode_image,
)

# Relatif à ce fichier de test, comme l'attend AppTest.from_file.
APP_SCRIPT = "../app.py"


def test_home_invites_to_draw_or_upload() -> None:
    app = AppTest.from_file(APP_SCRIPT).run()

    assert not app.exception
    assert app.title[0].value == SITE_TITLE
    assert "Tracez un chiffre" in app.info[0].value


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

    from dashboard.home import render_recognition

    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).line((50, 15, 50, 85), fill=0, width=10)
    render_recognition(recognize(load_model(DEFAULT_MODEL_PATH), image))


def test_recognition_shows_the_digit_and_its_probabilities() -> None:
    app = AppTest.from_function(render_a_recognized_digit).run()

    assert not app.exception
    assert app.metric[0].label == "Chiffre reconnu"
    assert app.caption[-1].value.startswith("Probabilités par chiffre")


@pytest.mark.parametrize(
    "payload", [b"pas une image", b"\x89PNG" + bytes(MAX_IMAGE_BYTES)], ids=["illisible", "lourde"]
)
def test_refused_images_raise_a_readable_error(payload: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode_image(payload)


def test_garbled_drawing_is_refused() -> None:
    with pytest.raises(InvalidImageError):
        decode_data_url(DATA_URL_PREFIX + "%%%")
