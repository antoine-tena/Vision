"""Activité de tout le réseau pour une image, et l'ordre dans lequel elle se propage.

Les neurones sont numérotés d'un bout à l'autre : les 784 pixels d'entrée, puis les neurones
cachés, puis les 10 sorties. Une étape de la propagation est le nombre de neurones déjà atteints.
"""

from dataclasses import dataclass

import numpy as np

from network.activations import softmax
from network.arrays import Labels, Matrix
from network.model import DigitClassifier
from pipeline.reasoning import hidden_activations, neuron_votes

INPUT_STEPS = 4
HIDDEN_STEPS = 8


@dataclass(frozen=True)
class NetworkActivity:
    pixels: Matrix
    hidden: Matrix
    probabilities: Matrix
    votes: Matrix

    @property
    def digit(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def layer_sizes(self) -> tuple[int, int, int]:
        return len(self.pixels), len(self.hidden), len(self.probabilities)

    @property
    def neuron_count(self) -> int:
        return sum(self.layer_sizes)


def network_activity(model: DigitClassifier, pixels: Matrix) -> NetworkActivity:
    """Ce que chaque neurone renvoie ; `votes` : ceux des neurones cachés pour le chiffre prédit."""
    hidden = hidden_activations(model, pixels)
    probabilities = softmax(model.logits(pixels)).reshape(-1)
    digit = int(np.argmax(probabilities))
    return NetworkActivity(
        pixels.reshape(-1), hidden, probabilities, neuron_votes(model, hidden, digit)
    )


def propagation_schedule(
    layer_sizes: tuple[int, ...], steps_per_layer: tuple[int, ...] = (INPUT_STEPS, HIDDEN_STEPS, 1)
) -> Labels:
    """Nombre de neurones atteints à chaque étape : chaque couche se remplit en quelques étapes,
    l'une après l'autre ; la dernière étape atteint tout le réseau."""
    starts = np.concatenate([[0], np.cumsum(layer_sizes)[:-1]])
    reached = [
        start + np.linspace(0, size, steps + 1)[1:]
        for start, size, steps in zip(starts, layer_sizes, steps_per_layer, strict=True)
    ]
    schedule: Labels = np.round(np.concatenate(reached)).astype(np.int64)
    return schedule
