"""Images reçues du navigateur : URL de données PNG renvoyées par la zone de dessin."""

import base64
import io

from PIL import Image

DATA_URL_PREFIX = "data:image/png;base64,"


def decode_data_url(data_url: str) -> Image.Image:
    """Image PIL tirée d'une URL de données PNG."""
    payload = base64.b64decode(data_url.removeprefix(DATA_URL_PREFIX))
    return Image.open(io.BytesIO(payload))
