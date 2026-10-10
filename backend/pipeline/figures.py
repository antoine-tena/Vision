"""Figures Plotly : une fonction par figure, qui reçoit des résultats déjà calculés."""

import numpy as np
import plotly.graph_objects as go

from network.arrays import Labels, Matrix
from network.model import DIGIT_COUNT, IMAGE_SIZE
from network.training import EpochMetrics

# Palette Okabe-Ito, lisible par les daltoniens ([FIGURES-ACCESSIBLES]).
COLORBLIND_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
DIGITS = [str(digit) for digit in range(DIGIT_COUNT)]


def training_history_figure(history: list[EpochMetrics]) -> go.Figure:
    """Exactitude d'entraînement et de validation par époque ; la perte sur un second axe."""
    epochs = [metrics.epoch for metrics in history]
    figure = go.Figure()
    for index, (name, values) in enumerate(
        (
            ("Exactitude (entraînement)", [m.train_accuracy * 100 for m in history]),
            ("Exactitude (validation)", [m.validation_accuracy * 100 for m in history]),
        )
    ):
        figure.add_trace(
            go.Scatter(x=epochs, y=values, name=name, line={"color": COLORBLIND_PALETTE[index]})
        )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=[m.loss for m in history],
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


def confusion_matrix(labels: Labels, predictions: Labels) -> Matrix:
    """Nombre d'images de chaque chiffre (lignes) prédites comme chaque chiffre (colonnes)."""
    matrix = np.zeros((DIGIT_COUNT, DIGIT_COUNT))
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


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


def probabilities_figure(probabilities: Matrix) -> go.Figure:
    """Probabilité donnée à chaque chiffre pour une image ; la prédiction ressort en couleur."""
    row = probabilities.reshape(DIGIT_COUNT)
    predicted = int(np.argmax(row))
    colors = [
        COLORBLIND_PALETTE[1] if digit == predicted else COLORBLIND_PALETTE[0]
        for digit in range(DIGIT_COUNT)
    ]
    figure = go.Figure(go.Bar(x=DIGITS, y=row * 100, marker={"color": colors}))
    figure.update_layout(title="Probabilités", xaxis_title="Chiffre", yaxis_title="Probabilité (%)")
    return figure


def digit_image(pixels: Matrix) -> Matrix:
    """Une ligne de 784 pixels remise en image 28x28, pour l'afficher."""
    image: Matrix = pixels.reshape(IMAGE_SIZE, IMAGE_SIZE)
    return image
