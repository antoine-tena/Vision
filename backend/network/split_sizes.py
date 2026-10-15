"""Taille des jeux d'entraînement, de validation et de test, pour l'affichage."""

from dataclasses import dataclass

from network.evaluate import PRINTED_TEST_PER_DIGIT
from network.model import DIGIT_COUNT
from network.train import PRINTED_PER_DIGIT, VALIDATION_FRACTION

MNIST_TRAIN_SIZE = 60_000
MNIST_TEST_SIZE = 10_000


@dataclass(frozen=True)
class SplitSizes:
    training: int
    validation: int
    test: int

    @property
    def total(self) -> int:
        return self.training + self.validation + self.test

    def share(self, count: int) -> float:
        return count / self.total


def split_sizes() -> SplitSizes:
    """MNIST et imprimés : 90 % / 10 % du jeu d'apprentissage, plus le test à part."""
    learning = MNIST_TRAIN_SIZE + PRINTED_PER_DIGIT * DIGIT_COUNT
    validation = int(learning * VALIDATION_FRACTION)
    return SplitSizes(
        training=learning - validation,
        validation=validation,
        test=MNIST_TEST_SIZE + PRINTED_TEST_PER_DIGIT * DIGIT_COUNT,
    )
