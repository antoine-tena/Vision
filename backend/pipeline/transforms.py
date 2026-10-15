"""Prétraitement : toute image de chiffre ramenée au format de MNIST, et découpage des jeux.

Format MNIST : chiffre clair sur fond noir, tenu dans une boîte de 20x20 pixels, centré par son
centre de masse dans une image de 28x28, pixels entre 0 et 1.
"""

import numpy as np
from PIL import Image

from network.arrays import Labels, Matrix
from network.model import DIGIT_COUNT, IMAGE_SIZE, PIXEL_COUNT
from pipeline.dataset import DigitDataset

DIGIT_BOX_SIZE = 20
INK_THRESHOLD = 0.1
LIGHT_BACKGROUND_MEAN = 0.5
PIXEL_MAX = 255.0


def to_mnist_pixels(image: Image.Image) -> Matrix:
    """Une ligne de 784 pixels prête pour le modèle (fond clair inversé, chiffre recentré)."""
    gray = np.asarray(image.convert("L"), dtype=np.float64) / PIXEL_MAX
    # Image surtout claire (papier, capture) : inversée pour un chiffre clair sur fond noir.
    if gray.mean() > LIGHT_BACKGROUND_MEAN:
        gray = 1.0 - gray
    # Pixels d'encre ; s'il n'y en a aucun, l'image est vide.
    ink_rows, ink_columns = np.nonzero(gray > INK_THRESHOLD)
    if len(ink_rows) == 0:
        return np.zeros((1, PIXEL_COUNT))
    # Recadrage au plus près de l'encre, puis mise à 20x20 et centrage dans 28x28.
    cropped = gray[ink_rows.min() : ink_rows.max() + 1, ink_columns.min() : ink_columns.max() + 1]
    return _center_by_mass(_fit_in_box(cropped)).reshape(1, PIXEL_COUNT)


def _fit_in_box(cropped: Matrix) -> Matrix:
    """Mise à l'échelle qui garde les proportions, le plus grand côté valant 20 pixels."""
    height, width = cropped.shape
    ratio = DIGIT_BOX_SIZE / max(height, width)
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    resized = Image.fromarray((cropped * PIXEL_MAX).astype(np.uint8)).resize(
        size, Image.Resampling.LANCZOS
    )
    fitted: Matrix = np.asarray(resized, dtype=np.float64) / PIXEL_MAX
    return fitted


def _center_by_mass(digit: Matrix) -> Matrix:
    """Place le chiffre dans 28x28 de sorte que son centre de masse tombe au centre."""
    height, width = digit.shape
    # Centre de masse : moyenne des positions des pixels, pondérée par leur intensité.
    rows, columns = np.indices(digit.shape)
    total = digit.sum()
    center_row = (rows * digit).sum() / total if total else height / 2
    center_column = (columns * digit).sum() / total if total else width / 2
    # Décalage qui amène ce centre au milieu, sans faire sortir le chiffre du cadre.
    top = int(np.clip(round(IMAGE_SIZE / 2 - center_row), 0, IMAGE_SIZE - height))
    left = int(np.clip(round(IMAGE_SIZE / 2 - center_column), 0, IMAGE_SIZE - width))
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    canvas[top : top + height, left : left + width] = digit
    return canvas


def split_dataset(
    dataset: DigitDataset, validation_fraction: float, seed: int
) -> tuple[DigitDataset, DigitDataset]:
    """Découpage aléatoire mais reproductible en jeu d'entraînement et jeu de validation."""
    order = np.random.default_rng(seed).permutation(len(dataset))
    validation_size = int(len(dataset) * validation_fraction)
    validation, training = order[:validation_size], order[validation_size:]
    return (
        DigitDataset(dataset.images[training], dataset.labels[training]),
        DigitDataset(dataset.images[validation], dataset.labels[validation]),
    )


def confusion_matrix(labels: Labels, predictions: Labels) -> Matrix:
    """Nombre d'images de chaque chiffre (lignes) prédites comme chaque chiffre (colonnes)."""
    matrix = np.zeros((DIGIT_COUNT, DIGIT_COUNT))
    # Un 1 ajouté dans la case (vrai chiffre, chiffre prédit) de chaque image.
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def digit_image(pixels: Matrix) -> Matrix:
    """Une ligne de 784 pixels remise en image 28x28, bornée entre 0 et 1 pour l'afficher."""
    image: Matrix = np.clip(pixels.reshape(IMAGE_SIZE, IMAGE_SIZE), 0.0, 1.0)
    return image
