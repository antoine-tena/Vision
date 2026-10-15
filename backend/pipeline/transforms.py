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
    if gray.mean() > LIGHT_BACKGROUND_MEAN:
        gray = 1.0 - gray
    ink_rows, ink_columns = np.nonzero(gray > INK_THRESHOLD)
    if len(ink_rows) == 0:
        return np.zeros((1, PIXEL_COUNT))
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
    rows, columns = np.indices(digit.shape)
    total = digit.sum()
    center_row = (rows * digit).sum() / total if total else height / 2
    center_column = (columns * digit).sum() / total if total else width / 2
    top = int(np.clip(round(IMAGE_SIZE / 2 - center_row), 0, IMAGE_SIZE - height))
    left = int(np.clip(round(IMAGE_SIZE / 2 - center_column), 0, IMAGE_SIZE - width))
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    canvas[top : top + height, left : left + width] = digit
    return canvas


def split_indices(size: int, validation_fraction: float, seed: int) -> tuple[Labels, Labels]:
    """Indices d'entraînement et de validation : tirage aléatoire mais reproductible."""
    order = np.random.default_rng(seed).permutation(size).astype(np.int64)
    validation_size = int(size * validation_fraction)
    return order[validation_size:], order[:validation_size]


def split_dataset(
    dataset: DigitDataset, validation_fraction: float, seed: int
) -> tuple[DigitDataset, DigitDataset]:
    """Découpage en jeu d'entraînement et jeu de validation (voir `split_indices`)."""
    training, validation = split_indices(len(dataset), validation_fraction, seed)
    return dataset.subset(training), dataset.subset(validation)


def sample_per_digit(dataset: DigitDataset, per_digit: int, seed: int) -> DigitDataset:
    """`per_digit` images de chaque chiffre (moins s'il en manque), triées par chiffre."""
    rng = np.random.default_rng(seed)
    chosen = [
        rng.permutation(np.flatnonzero(dataset.labels == digit))[:per_digit]
        for digit in range(DIGIT_COUNT)
    ]
    return dataset.subset(np.concatenate(chosen).astype(np.int64))


def confusion_matrix(labels: Labels, predictions: Labels) -> Matrix:
    """Nombre d'images de chaque chiffre (lignes) prédites comme chaque chiffre (colonnes)."""
    matrix = np.zeros((DIGIT_COUNT, DIGIT_COUNT))
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def confusion_accuracy(matrix: Matrix) -> float:
    """Part des images bien reconnues (diagonale) ; 0 pour une matrice vide."""
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else 0.0


def digit_image(pixels: Matrix) -> Matrix:
    """Une ligne de 784 pixels remise en image 28x28, bornée entre 0 et 1 pour l'afficher."""
    image: Matrix = np.clip(pixels.reshape(IMAGE_SIZE, IMAGE_SIZE), 0.0, 1.0)
    return image


def top_confusions(matrix: Matrix, count: int) -> list[tuple[int, int, int]]:
    """Les `count` erreurs les plus fréquentes : (vrai chiffre, réponse, nombre de fois)."""
    errors = matrix.astype(np.int64).copy()
    np.fill_diagonal(errors, 0)
    order = np.argsort(-errors, axis=None)[:count]
    rows, columns = np.unravel_index(order, errors.shape)
    return [
        (int(row), int(column), int(errors[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if errors[row, column] > 0
    ]
