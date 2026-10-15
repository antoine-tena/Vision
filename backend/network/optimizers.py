"""Descente de gradient stochastique avec inertie et décroissance du pas (`Optimizer_SGD`)."""

from network.layers import DenseLayer


class StochasticGradientDescent:
    def __init__(self, learning_rate: float, decay: float = 0.0, momentum: float = 0.0) -> None:
        """Règle l'optimiseur.

        learning_rate : taille du pas de départ ; decay : vitesse à laquelle le pas rétrécit ;
        momentum : part du déplacement précédent conservée (0 = pas d'inertie).
        """
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
            # Déplacement = inertie x déplacement précédent - pas x gradient : on descend la
            # pente de la perte en gardant de l'élan, ce qui lisse et accélère la descente.
            layer.weight_momentums = (
                self.momentum * layer.weight_momentums - rate * layer.weight_gradients
            )
            layer.bias_momentums = (
                self.momentum * layer.bias_momentums - rate * layer.bias_gradients
            )
            layer.weights += layer.weight_momentums
            layer.biases += layer.bias_momentums
        self.iterations += 1
