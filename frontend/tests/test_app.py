import base64
import io

from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from dashboard.constants import SITE_TITLE
from dashboard.images import DATA_URL_PREFIX, decode_data_url

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
