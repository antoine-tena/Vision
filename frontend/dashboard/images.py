"""Images reçues du navigateur : validées à la frontière ([VALIDATION-AUX-FRONTIERES]).

Dessin (URL de données PNG renvoyée par la zone de dessin) ou fichier importé : tout passe par
`decode_image`, qui refuse un fichier trop lourd, illisible ou démesuré avec un message clair.
"""

import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError

DATA_URL_PREFIX = "data:image/png;base64,"
BYTES_PER_MEGABYTE = 1024 * 1024
MAX_IMAGE_MEGABYTES = 5
MAX_IMAGE_BYTES = MAX_IMAGE_MEGABYTES * BYTES_PER_MEGABYTE
MAX_IMAGE_SIDE = 4096


class InvalidImageError(ValueError):
    """Image refusée ; le message s'affiche tel quel à l'utilisateur."""


def decode_image(payload: bytes) -> Image.Image:
    """Image PIL entièrement lue, ou InvalidImageError si elle est refusée."""
    # Poids vérifié avant toute lecture, dimensions avant de décoder les pixels.
    if len(payload) > MAX_IMAGE_BYTES:
        raise InvalidImageError(
            f"L'image dépasse {MAX_IMAGE_MEGABYTES} Mo : choisissez un fichier plus léger."
        )
    try:
        image = Image.open(io.BytesIO(payload))
        if max(image.size) > MAX_IMAGE_SIDE:
            raise InvalidImageError(
                f"L'image est trop grande ({MAX_IMAGE_SIDE} pixels de côté au plus)."
            )
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImageError("Ce fichier n'est pas une image lisible.") from error
    return image


def decode_data_url(data_url: str) -> Image.Image:
    """Image PIL tirée d'une URL de données PNG."""
    try:
        payload = base64.b64decode(data_url.removeprefix(DATA_URL_PREFIX), validate=True)
    except binascii.Error as error:
        raise InvalidImageError("Le dessin reçu est illisible : effacez et recommencez.") from error
    return decode_image(payload)
