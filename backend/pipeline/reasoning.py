"""Raisonnement interne du réseau pour une image : ce qui l'a poussé vers un chiffre.

Le réseau est linéaire par morceaux : une fois connus les neurones cachés allumés (ReLU > 0), le
score d'un chiffre est exactement la somme des votes de ces neurones (activation * poids vers le
chiffre), plus les biais. On en tire deux lectures :
- les neurones qui ont le plus voté, pour ou contre, et le motif de pixels que chacun guette ;
- une carte des pixels décisifs : pixel * gradient du score, ce que chaque pixel a apporté.
"""

from dataclasses import dataclass

import numpy as np

from network.arrays import Labels, Matrix
from network.model import IMAGE_SIZE, DigitClassifier
from pipeline.neuron_shapes import active_shapes, shape_name

TOP_NEURONS = 8


@dataclass(frozen=True)
class Reasoning:
    """L'explication du score d'un chiffre ; les tableaux `neuron_*` vont du plus influent au
    moins influent (valeur absolue du vote)."""

    digit: int
    score: float
    active_count: int
    hidden_size: int
    evidence: Matrix
    neuron_ids: Labels
    neuron_activations: Matrix
    neuron_votes: Matrix
    neuron_patterns: Matrix
    neuron_shapes: list[str]
    active_shapes: list[tuple[str, int]]


def hidden_activations(model: DigitClassifier, pixels: Matrix) -> Matrix:
    """Sortie de la couche cachée (après ReLU) pour une ligne de 784 pixels."""
    activations: Matrix = np.maximum(0.0, pixels @ model.hidden.weights + model.hidden.biases)
    return activations.reshape(-1)


def neuron_votes(model: DigitClassifier, activations: Matrix, digit: int) -> Matrix:
    """Vote de chaque neurone caché pour le chiffre : activation * poids vers ce chiffre."""
    votes: Matrix = activations * model.output.weights[:, digit]
    return votes


def pixel_evidence(model: DigitClassifier, pixels: Matrix, digit: int) -> Matrix:
    """Apport de chaque pixel au score du chiffre (pixel * gradient), en image 28x28."""
    is_active = hidden_activations(model, pixels) > 0
    gradient = model.hidden.weights[:, is_active] @ model.output.weights[is_active, digit]
    evidence: Matrix = (pixels.reshape(-1) * gradient).reshape(IMAGE_SIZE, IMAGE_SIZE)
    return evidence


def explain(
    model: DigitClassifier, pixels: Matrix, digit: int, top_neurons: int = TOP_NEURONS
) -> Reasoning:
    activations = hidden_activations(model, pixels)
    votes = neuron_votes(model, activations, digit)
    ranked = np.argsort(-np.abs(votes))[:top_neurons].astype(np.int64)
    patterns = neuron_patterns(model, np.arange(len(activations)))
    return Reasoning(
        digit=digit,
        score=float(votes.sum() + model.output.biases[0, digit]),
        active_count=int((activations > 0).sum()),
        hidden_size=len(activations),
        evidence=pixel_evidence(model, pixels, digit),
        neuron_ids=ranked,
        neuron_activations=activations[ranked],
        neuron_votes=votes[ranked],
        neuron_patterns=patterns[ranked],
        neuron_shapes=[shape_name(pattern) for pattern in patterns[ranked]],
        active_shapes=active_shapes(patterns, activations),
    )


def neuron_patterns(model: DigitClassifier, neuron_ids: Labels) -> Matrix:
    """Poids d'entrée des neurones, chacun en image 28x28 : le motif qu'il guette."""
    patterns: Matrix = model.hidden.weights[:, neuron_ids].T.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    return patterns


def hidden_patterns_mosaic(model: DigitClassifier, columns: int) -> Matrix:
    """Les motifs de tous les neurones cachés, en mosaïque de `columns` colonnes."""
    hidden_size = model.hidden.weights.shape[1]
    return tile_patterns(neuron_patterns(model, np.arange(hidden_size)), columns)


def tile_patterns(patterns: Matrix, columns: int) -> Matrix:
    """Motifs 28x28 rangés en mosaïque, séparés par une ligne vide (NaN, non tracée)."""
    rows = -(-len(patterns) // columns)
    cell = IMAGE_SIZE + 1
    padded = np.pad(
        patterns.astype(np.float64),
        ((0, rows * columns - len(patterns)), (0, 1), (0, 1)),
        constant_values=np.nan,
    )
    grid = padded.reshape(rows, columns, cell, cell).transpose(0, 2, 1, 3)
    mosaic: Matrix = grid.reshape(rows * cell, columns * cell)[:-1, :-1]
    return mosaic


def evidence_totals(evidence: Matrix) -> tuple[float, float]:
    """Apport total des pixels pour le chiffre et contre lui (les deux positifs)."""
    return float(evidence[evidence > 0].sum()), float(-evidence[evidence < 0].sum())
