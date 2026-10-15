"""Figures Plotly : une fonction par figure, qui reçoit des résultats déjà calculés."""

import numpy as np
import plotly.graph_objects as go

from network.arrays import Matrix
from network.model import DIGIT_COUNT
from network.training import EpochMetrics

# Palette Okabe-Ito, lisible par les daltoniens ([FIGURES-ACCESSIBLES]).
COLORBLIND_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
DIGITS = [str(digit) for digit in range(DIGIT_COUNT)]
PREDICTED_PATTERN = "/"
TOP_DIGITS_DESCRIBED = 3


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


def confusion_matrix_figure(matrix: Matrix) -> go.Figure:
    """Carte de chaleur : ligne = chiffre réel, colonne = chiffre prédit, case = nombre d'images.

    La diagonale compte les bonnes réponses ; une case forte hors diagonale montre une confusion
    fréquente. Le nombre est écrit dans chaque case : l'information ne tient pas qu'à la couleur.
    """
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
    """Probabilité donnée à chaque chiffre pour une image ; la prédiction ressort par sa
    couleur, un motif hachuré et son pourcentage écrit ([FIGURES-ACCESSIBLES])."""
    row = probabilities.reshape(DIGIT_COUNT)
    predicted = int(np.argmax(row))
    is_predicted = [digit == predicted for digit in range(DIGIT_COUNT)]
    figure = go.Figure(
        go.Bar(
            x=DIGITS,
            y=row * 100,
            marker={
                "color": [COLORBLIND_PALETTE[1 if flag else 0] for flag in is_predicted],
                "pattern": {"shape": [PREDICTED_PATTERN if flag else "" for flag in is_predicted]},
            },
            text=[f"{row[predicted]:.0%} (prédit)" if flag else "" for flag in is_predicted],
            textposition="outside",
        )
    )
    figure.update_layout(title="Probabilités", xaxis_title="Chiffre", yaxis_title="Probabilité (%)")
    return figure


def probabilities_description(probabilities: Matrix) -> str:
    """Texte alternatif de la figure des probabilités : ce qu'elle montre, en une phrase."""
    row = probabilities.reshape(DIGIT_COUNT)
    ranked = np.argsort(row)[::-1][:TOP_DIGITS_DESCRIBED]
    parts = ", ".join(f"{digit} à {row[digit]:.0%}" for digit in ranked)
    return f"Probabilités par chiffre ; les plus probables : {parts}."
