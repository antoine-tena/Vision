"""Fonctions d'activation : ReLU (`Activation_ReLU`) et softmax (`Activation_Softmax`)."""

import numpy as np

from network.arrays import Matrix


class ReLU:
    """max(0, x) ; le gradient ne passe que là où l'entrée était positive."""

    def __init__(self) -> None:
        self._inputs: Matrix = np.zeros((0, 0))

    def forward(self, inputs: Matrix) -> Matrix:
        self._inputs = inputs
        outputs: Matrix = np.maximum(0.0, inputs)
        return outputs

    def backward(self, output_gradients: Matrix) -> Matrix:
        input_gradients: Matrix = np.where(self._inputs > 0, output_gradients, 0.0)
        return input_gradients


def softmax(logits: Matrix) -> Matrix:
    """Probabilités par ligne ; le maximum est retiré avant l'exponentielle (stabilité)."""
    exponentials = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities: Matrix = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return probabilities
