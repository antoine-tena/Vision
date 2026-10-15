"""Fonctions d'activation : ReLU (`Activation_ReLU`) et softmax (`Activation_Softmax`)."""

import numpy as np

from network.arrays import Matrix


class ReLU:
    """max(0, x) ; le gradient ne passe que là où l'entrée était positive."""

    def __init__(self) -> None:
        """Prépare la mémoire des entrées, dont la rétropropagation a besoin."""
        self._inputs: Matrix = np.zeros((0, 0))

    def forward(self, inputs: Matrix) -> Matrix:
        """Propagation : les valeurs négatives deviennent 0, les positives passent telles quelles.

        Les entrées sont gardées pour savoir, au retour, quels neurones étaient actifs.
        """
        self._inputs = inputs
        outputs: Matrix = np.maximum(0.0, inputs)
        return outputs

    def backward(self, output_gradients: Matrix) -> Matrix:
        """Rétropropagation : le gradient passe là où l'entrée était positive, 0 ailleurs.

        La dérivée de max(0, x) vaut 1 si x > 0 et 0 sinon.
        """
        input_gradients: Matrix = np.where(self._inputs > 0, output_gradients, 0.0)
        return input_gradients


def softmax(logits: Matrix) -> Matrix:
    """Transforme les scores (logits) en probabilités : chaque ligne est positive et somme à 1.

    softmax(z)_i = exp(z_i) / somme des exp(z_j). Le maximum de la ligne est retiré avant
    l'exponentielle : le résultat est le même, mais exp ne déborde pas sur de grands scores.
    """
    exponentials = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities: Matrix = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return probabilities
