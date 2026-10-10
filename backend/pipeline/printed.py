"""Chiffres imprimés (« numériques ») générés à partir des polices du poste.

MNIST ne contient que des chiffres manuscrits : pour reconnaître aussi des chiffres tapés, on
dessine chaque chiffre dans des polices variées, avec une taille, une inclinaison et une
épaisseur tirées au hasard (graine fixée, [REPRODUCTIBLE]), puis on le ramène au format MNIST.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from network.arrays import Labels
from pipeline.dataset import DigitDataset
from pipeline.transforms import to_mnist_pixels

FONT_DIRS = (
    Path("/usr/share/fonts"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path("/mnt/c/Windows/Fonts"),
)
FONT_SUFFIXES = (".ttf", ".otf")
# Polices d'icônes ou de symboles : pas de vrais chiffres.
EXCLUDED_FONT_WORDS = ("awesome", "symbol", "emoji", "math", "dingbat", "wingding", "icon")
MAX_FONTS = 60
CANVAS_SIZE = 96
FONT_SIZE_RANGE = (48, 72)
ROTATION_DEGREES = 12.0
MAX_STROKE_WIDTH = 2
FALLBACK_FONT_SIZE = 64


def find_fonts(font_dirs: tuple[Path, ...] = FONT_DIRS) -> list[Path]:
    """Polices du poste, triées pour que la génération soit la même d'un lancement à l'autre."""
    fonts = sorted(
        path
        for directory in font_dirs
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.suffix.lower() in FONT_SUFFIXES
        and not any(word in path.name.lower() for word in EXCLUDED_FONT_WORDS)
    )
    # Une sur n, pour couvrir toutes les familles plutôt que les premières par ordre alphabétique.
    return fonts[:: max(1, len(fonts) // MAX_FONTS)][:MAX_FONTS]


def render_digit(digit: int, font_path: Path | None, rng: np.random.Generator) -> Image.Image:
    """Le chiffre en blanc sur fond noir, dans une police, une taille et un angle tirés."""
    size = int(rng.integers(*FONT_SIZE_RANGE))
    font = (
        ImageFont.truetype(str(font_path), size)
        if font_path
        else ImageFont.load_default(size=FALLBACK_FONT_SIZE)
    )
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    stroke_width = int(rng.integers(0, MAX_STROKE_WIDTH + 1))
    ImageDraw.Draw(canvas).text(
        (CANVAS_SIZE / 2, CANVAS_SIZE / 2),
        str(digit),
        fill=255,
        font=font,
        anchor="mm",
        stroke_width=stroke_width,
        stroke_fill=255,
    )
    angle = float(rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES))
    return canvas.rotate(angle, resample=Image.Resampling.BILINEAR)


def generate_printed_digits(
    count_per_digit: int, seed: int, fonts: list[Path] | None = None
) -> DigitDataset:
    """`count_per_digit` images de chaque chiffre, polices tirées parmi celles du poste."""
    rng = np.random.default_rng(seed)
    available: list[Path | None] = list(fonts if fonts is not None else find_fonts()) or [None]
    images, labels = [], []
    for digit in range(10):
        for _ in range(count_per_digit):
            font_path = available[int(rng.integers(len(available)))]
            images.append(to_mnist_pixels(render_digit(digit, font_path, rng))[0])
            labels.append(digit)
    label_array: Labels = np.array(labels, dtype=np.int64)
    return DigitDataset(np.array(images), label_array)
