"""Descente de gradient stochastique avec inertie et décroissance du pas (`Optimizer_SGD`)."""

from network.layers import DenseLayer


class StochasticGradientDescent:
    def __init__(self, learning_rate: float, decay: float = 0.0, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    @property
    def current_learning_rate(self) -> float:
        """Pas qui décroît avec les itérations : lr / (1 + décroissance x itérations)."""
        return self.learning_rate / (1.0 + self.decay * self.iterations)

    def step(self, layers: list[DenseLayer]) -> None:
        """Met à jour chaque couche à partir de ses gradients, puis compte l'itération."""
        rate = self.current_learning_rate
        for layer in layers:
            layer.weight_momentums = (
                self.momentum * layer.weight_momentums - rate * layer.weight_gradients
            )
            layer.bias_momentums = (
                self.momentum * layer.bias_momentums - rate * layer.bias_gradients
            )
            layer.weights += layer.weight_momentums
            layer.biases += layer.bias_momentums
        self.iterations += 1
