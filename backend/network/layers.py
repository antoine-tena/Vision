"""Couche dense : sortie = entrées · poids + biais."""

import numpy as np

from network.arrays import Matrix


class DenseLayer:
    """Couche entièrement connectée, avec ses gradients et l'inertie de son optimiseur."""

    def __init__(self, input_size: int, output_size: int, rng: np.random.Generator) -> None:
        """Poids tirés au hasard, biais, gradients et inerties à zéro.

        `weights` a une ligne par entrée et une colonne par neurone de la couche.
        """
        # Initialisation de He : l'échelle suit le nombre d'entrées, adaptée à ReLU.
        scale = np.sqrt(2.0 / input_size)
        self.weights: Matrix = scale * rng.standard_normal((input_size, output_size))
        self.biases: Matrix = np.zeros((1, output_size))
        self.weight_gradients: Matrix = np.zeros_like(self.weights)
        self.bias_gradients: Matrix = np.zeros_like(self.biases)
        self.weight_momentums: Matrix = np.zeros_like(self.weights)
        self.bias_momentums: Matrix = np.zeros_like(self.biases)
        self._inputs: Matrix = np.zeros((0, input_size))

    def forward(self, inputs: Matrix) -> Matrix:
        """Propagation : chaque neurone fait la somme pondérée des entrées, plus son biais.

        Avec n images en lignes : (n, entrées) @ (entrées, sorties) + (1, sorties) → (n, sorties).
        Les entrées sont gardées pour calculer le gradient des poids au retour.
        """
        self._inputs = inputs
        outputs: Matrix = inputs @ self.weights + self.biases
        return outputs

    def backward(self, output_gradients: Matrix) -> Matrix:
        """Rétropropagation : reçoit dPerte/dSortie, renvoie dPerte/dEntrée.

        Les gradients des poids et des biais sont gardés pour l'optimiseur ; celui des entrées
        remonte à la couche précédente (dérivation en chaîne).
        """
        # dPerte/dPoids = entréesᵀ · dPerte/dSortie, sommé sur les images du lot.
        self.weight_gradients = self._inputs.T @ output_gradients
        # Un biais sert à toutes les images : son gradient est la somme sur le lot.
        self.bias_gradients = np.sum(output_gradients, axis=0, keepdims=True)
        # dPerte/dEntrée = dPerte/dSortie · poidsᵀ : ce que la couche précédente doit corriger.
        input_gradients: Matrix = output_gradients @ self.weights.T
        return input_gradients
