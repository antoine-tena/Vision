"""Reconnaissance d'une image quelconque : prétraitement au format MNIST, puis prédiction."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from network.arrays import Matrix
from network.model import DigitClassifier
from pipeline.transforms import to_mnist_pixels


@dataclass(frozen=True)
class Recognition:
    digit: int
    confidence: float
    probabilities: Matrix
    pixels: Matrix

    @property
    def is_blank(self) -> bool:
        return not self.pixels.any()


def recognize(model: DigitClassifier, image: Image.Image) -> Recognition:
    pixels = to_mnist_pixels(image)
    probabilities = model.predict_proba(pixels)
    digit = int(np.argmax(probabilities))
    return Recognition(digit, float(probabilities[0, digit]), probabilities, pixels)
