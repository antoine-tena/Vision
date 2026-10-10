"""Couche dense : sortie = entrées · poids + biais."""

import numpy as np

from network.arrays import Matrix


class DenseLayer:
    """Couche entièrement connectée, avec ses gradients et l'inertie de son optimiseur."""

    def __init__(self, input_size: int, output_size: int, rng: np.random.Generator) -> None:
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
        self._inputs = inputs
        outputs: Matrix = inputs @ self.weights + self.biases
        return outputs

    def backward(self, output_gradients: Matrix) -> Matrix:
        """Gradients des poids et des biais, puis gradient renvoyé à la couche précédente."""
        self.weight_gradients = self._inputs.T @ output_gradients
        self.bias_gradients = np.sum(output_gradients, axis=0, keepdims=True)
        input_gradients: Matrix = output_gradients @ self.weights.T
        return input_gradients
