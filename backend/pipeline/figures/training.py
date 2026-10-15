"""Figures de l'entraînement : courbes, matrice de confusion, motifs appris."""

import numpy as np
import plotly.graph_objects as go

from network.arrays import Matrix
from network.model import DIGIT_COUNT
from network.training import EpochMetrics
from pipeline.figures.common import (
    COLORBLIND_PALETTE,
    DIGITS,
    DIVERGING_COLORSCALE,
)
from pipeline.transforms import confusion_accuracy


def training_history_figure(history: list[EpochMetrics]) -> go.Figure:
    """Exactitude d'entraînement et de validation par époque ; la perte sur un second axe."""
    epochs = [metrics.epoch for metrics in history]
    figure = go.Figure()
    for index, (name, values) in enumerate(
        (
            ("Exactitude (entraînement)", [metrics.train_accuracy * 100 for metrics in history]),
            ("Exactitude (validation)", [metrics.validation_accuracy * 100 for metrics in history]),
        )
    ):
        figure.add_trace(
            go.Scatter(x=epochs, y=values, name=name, line={"color": COLORBLIND_PALETTE[index]})
        )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=[metrics.loss for metrics in history],
            name="Perte",
            yaxis="y2",
            line={"color": COLORBLIND_PALETTE[2], "dash": "dot"},
        )
    )
    figure.update_layout(
        title="Apprentissage",
        xaxis_title="Époque",
        yaxis_title="Exactitude (%)",
        yaxis2={"title": "Perte (entropie croisée)", "overlaying": "y", "side": "right"},
    )
    return figure


def training_history_description(history: list[EpochMetrics]) -> str:
    first, last = history[0], history[-1]
    return (
        f"Sur {len(history)} époques, la perte passe de {first.loss:.3f} à {last.loss:.3f} ; "
        f"l'exactitude de validation de {first.validation_accuracy:.1%} "
        f"à {last.validation_accuracy:.1%}."
    )


def confusion_matrix_figure(matrix: Matrix) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=DIGITS,
            y=DIGITS,
            colorscale="Cividis",
            text=matrix.astype(int),
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        title="Matrice de confusion",
        xaxis_title="Chiffre prédit",
        yaxis_title="Chiffre réel",
        yaxis={"autorange": "reversed"},
    )
    return figure


def confusion_description(matrix: Matrix) -> str:
    """Texte alternatif de la matrice de confusion : exactitude et confusion la plus fréquente."""
    errors = matrix * (1 - np.eye(DIGIT_COUNT))
    real, predicted = np.unravel_index(int(np.argmax(errors)), errors.shape)
    error_count = int(errors[real, predicted])
    return (
        f"{confusion_accuracy(matrix):.1%} de {int(matrix.sum())} images bien reconnues ; "
        f"confusion la plus fréquente : {real} pris pour {predicted} ({error_count} fois)."
    )


def hidden_patterns_figure(mosaic: Matrix) -> go.Figure:
    """Mosaïque des motifs appris par les neurones de la couche cachée."""
    limit = float(np.nanmax(np.abs(mosaic))) if np.isfinite(mosaic).any() else 1.0
    figure = go.Figure(
        go.Heatmap(
            z=mosaic,
            colorscale=DIVERGING_COLORSCALE,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "Poids"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(title="Ce que guette chaque neurone caché (poids d'entrée)")
    figure.update_xaxes(showticklabels=False, constrain="domain")
    figure.update_yaxes(showticklabels=False, autorange="reversed", scaleanchor="x")
    return figure
