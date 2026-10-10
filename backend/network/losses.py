"""Softmax suivie de l'entropie croisée (`Activation_Softmax_Loss_CategoricalCrossentropy`).

Les deux étapes sont réunies parce que leur gradient combiné est simple et stable :
(probabilités - classes attendues en one-hot) / nombre d'exemples.
"""

import numpy as np

from network.activations import softmax
from network.arrays import Labels, Matrix

PROBABILITY_FLOOR = 1e-7


def cross_entropy(probabilities: Matrix, labels: Labels) -> float:
    """Perte moyenne : -log de la probabilité donnée à la bonne classe."""
    clipped = np.clip(probabilities, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    correct_confidences = clipped[np.arange(len(labels)), labels]
    return float(np.mean(-np.log(correct_confidences)))


class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self.probabilities: Matrix = np.zeros((0, 0))
        self._labels: Labels = np.zeros(0, dtype=np.int64)

    def forward(self, logits: Matrix, labels: Labels) -> float:
        self.probabilities = softmax(logits)
        self._labels = labels
        return cross_entropy(self.probabilities, labels)

    def backward(self) -> Matrix:
        gradients = self.probabilities.copy()
        gradients[np.arange(len(self._labels)), self._labels] -= 1.0
        logit_gradients: Matrix = gradients / len(self._labels)
        return logit_gradients
