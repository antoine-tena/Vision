"""Softmax suivie de l'entropie croisée (`Activation_Softmax_Loss_CategoricalCrossentropy`).

Les deux étapes sont réunies parce que leur gradient combiné est simple et stable :
(probabilités - classes attendues en one-hot) / nombre d'exemples.
"""

import numpy as np

from network.activations import softmax
from network.arrays import Labels, Matrix

PROBABILITY_FLOOR = 1e-7


def cross_entropy(probabilities: Matrix, labels: Labels) -> float:
    """Perte moyenne du lot : -log de la probabilité donnée à la bonne classe.

    Bonne réponse sûre (probabilité 1) → perte 0 ; bonne classe jugée improbable → perte forte.
    Les probabilités sont bornées loin de 0 pour que log(0) ne donne pas l'infini.
    """
    clipped = np.clip(probabilities, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    # Pour chaque image, la probabilité donnée à son vrai chiffre.
    correct_confidences = clipped[np.arange(len(labels)), labels]
    return float(np.mean(-np.log(correct_confidences)))


class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        """Prépare la mémoire des probabilités et des chiffres attendus, utiles au retour."""
        self.probabilities: Matrix = np.zeros((0, 0))
        self._labels: Labels = np.zeros(0, dtype=np.int64)

    def forward(self, logits: Matrix, labels: Labels) -> float:
        """Propagation : scores → probabilités (softmax) → perte (entropie croisée)."""
        self.probabilities = softmax(logits)
        self._labels = labels
        return cross_entropy(self.probabilities, labels)

    def backward(self) -> Matrix:
        """Gradient de la perte par rapport aux scores : (probabilités - one-hot) / n.

        Retirer 1 à la probabilité du bon chiffre revient à soustraire le vecteur one-hot ;
        diviser par n donne le gradient de la perte moyenne.
        """
        gradients = self.probabilities.copy()
        gradients[np.arange(len(self._labels)), self._labels] -= 1.0
        logit_gradients: Matrix = gradients / len(self._labels)
        return logit_gradients
