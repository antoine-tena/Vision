"""Figures Plotly : une fonction par figure, qui reçoit des résultats déjà calculés."""

from pipeline.figures.common import COLORBLIND_PALETTE, DIGITS, DIVERGING_COLORSCALE
from pipeline.figures.network import network_description, network_figure
from pipeline.figures.recognition import (
    evidence_description,
    evidence_figure,
    neurons_description,
    neurons_figure,
    output_weights_figure,
    patterns_figure,
    probabilities_description,
    probabilities_figure,
    scores_figure,
)
from pipeline.figures.training import (
    confusion_description,
    confusion_matrix_figure,
    hidden_patterns_figure,
    training_history_description,
    training_history_figure,
)

__all__ = [
    "COLORBLIND_PALETTE",
    "DIGITS",
    "DIVERGING_COLORSCALE",
    "confusion_description",
    "confusion_matrix_figure",
    "evidence_description",
    "evidence_figure",
    "hidden_patterns_figure",
    "network_description",
    "network_figure",
    "neurons_description",
    "neurons_figure",
    "output_weights_figure",
    "patterns_figure",
    "probabilities_description",
    "probabilities_figure",
    "scores_figure",
    "training_history_description",
    "training_history_figure",
]
