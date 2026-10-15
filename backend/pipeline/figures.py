"""Figures Plotly : une fonction par figure, qui reçoit des résultats déjà calculés."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from network.arrays import Matrix
from network.model import DIGIT_COUNT
from network.training import EpochMetrics
from pipeline.reasoning import Reasoning
from pipeline.transforms import confusion_accuracy

# Palette Okabe-Ito, lisible par les daltoniens ([FIGURES-ACCESSIBLES]).
COLORBLIND_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
DIGITS = [str(digit) for digit in range(DIGIT_COUNT)]
PREDICTED_PATTERN = "/"
TOP_DIGITS_DESCRIBED = 3
# Divergente : bleu = pousse vers le chiffre, orange = l'en éloigne, blanc = neutre.
DIVERGING_COLORSCALE = [
    [0.0, COLORBLIND_PALETTE[5]],
    [0.5, "#FFFFFF"],
    [1.0, COLORBLIND_PALETTE[0]],
]
NEURON_GRID_COLUMNS = 4


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


def training_history_description(history: list[EpochMetrics]) -> str:
    first, last = history[0], history[-1]
    return (
        f"Sur {len(history)} époques, la perte passe de {first.loss:.3f} à {last.loss:.3f} ; "
        f"l'exactitude de validation de {first.validation_accuracy:.1%} "
        f"à {last.validation_accuracy:.1%}."
    )


def confusion_description(matrix: Matrix) -> str:
    """Texte alternatif de la matrice de confusion : exactitude et confusion la plus fréquente."""
    errors = matrix * (1 - np.eye(DIGIT_COUNT))
    real, predicted = np.unravel_index(int(np.argmax(errors)), errors.shape)
    error_count = int(errors[real, predicted])
    return (
        f"{confusion_accuracy(matrix):.1%} de {int(matrix.sum())} images bien reconnues ; "
        f"confusion la plus fréquente : {real} pris pour {predicted} ({error_count} fois)."
    )


def evidence_figure(evidence: Matrix, digit: int) -> go.Figure:
    """Carte des pixels décisifs : bleu pour le chiffre, orange contre, valeur au survol."""
    limit = float(np.abs(evidence).max()) or 1.0
    figure = go.Figure(
        go.Heatmap(
            z=evidence,
            colorscale=DIVERGING_COLORSCALE,
            zmin=-limit,
            zmax=limit,
            colorbar={
                "title": "Apport",
                "tickvals": [-limit, 0, limit],
                "ticktext": ["contre", "0", "pour"],
            },
            hovertemplate="ligne %{y}, colonne %{x}<br>apport %{z:+.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Pixels qui comptent pour le {digit}",
        xaxis={"title": "Colonne (pixel)", "constrain": "domain"},
        yaxis={"title": "Ligne (pixel)", "autorange": "reversed", "scaleanchor": "x"},
    )
    return figure


def evidence_description(evidence: Matrix, digit: int) -> str:
    support, against = evidence[evidence > 0].sum(), -evidence[evidence < 0].sum()
    return (
        f"Carte des pixels décisifs pour le {digit} : apport total {support:.2f} pour, "
        f"{against:.2f} contre ; en bleu les traits qui ressemblent au {digit}, en orange ceux "
        "qui l'en éloignent."
    )


def neurons_figure(reasoning: Reasoning) -> go.Figure:
    """Motif guetté par chacun des neurones cachés qui ont le plus voté, avec son vote."""
    count = len(reasoning.neuron_ids)
    rows = -(-count // NEURON_GRID_COLUMNS)
    titles = [
        f"n°{neuron} : {vote:+.2f}"
        for neuron, vote in zip(reasoning.neuron_ids, reasoning.neuron_votes, strict=True)
    ]
    figure = make_subplots(
        rows=rows,
        cols=NEURON_GRID_COLUMNS,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.12,
    )
    for index, pattern in enumerate(reasoning.neuron_patterns):
        figure.add_trace(
            go.Heatmap(
                z=pattern, coloraxis="coloraxis", hovertemplate="poids %{z:+.3f}<extra></extra>"
            ),
            row=index // NEURON_GRID_COLUMNS + 1,
            col=index % NEURON_GRID_COLUMNS + 1,
        )
    limit = float(np.abs(reasoning.neuron_patterns).max()) if count else 1.0
    figure.update_layout(
        title=f"Neurones qui ont le plus voté pour ou contre le {reasoning.digit}",
        coloraxis={
            "colorscale": DIVERGING_COLORSCALE,
            "cmin": -limit,
            "cmax": limit,
            "colorbar": {"title": "Poids"},
        },
    )
    figure.update_xaxes(showticklabels=False, constrain="domain")
    figure.update_yaxes(showticklabels=False, autorange="reversed", scaleanchor="x")
    return figure


def neurons_description(reasoning: Reasoning) -> str:
    votes = ", ".join(
        f"n°{neuron} {vote:+.2f}"
        for neuron, vote in zip(reasoning.neuron_ids, reasoning.neuron_votes, strict=True)
    )
    return (
        f"Motifs des neurones cachés les plus influents et leur vote pour le {reasoning.digit} : "
        f"{votes}. En bleu les pixels où le neurone attend de l'encre, en orange où il n'en "
        "veut pas."
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
