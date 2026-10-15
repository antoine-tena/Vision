"""Figures d'une reconnaissance : probabilités, pixels décisifs, neurones influents."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from network.arrays import Matrix
from network.model import DIGIT_COUNT
from pipeline.figures.common import (
    COLORBLIND_PALETTE,
    DIGITS,
    DIVERGING_COLORSCALE,
)
from pipeline.reasoning import Reasoning

PREDICTED_PATTERN = "/"
TOP_DIGITS_DESCRIBED = 3
NEURON_GRID_COLUMNS = 4


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
    figure.update_xaxes(type="category")
    return figure


def probabilities_description(probabilities: Matrix) -> str:
    """Texte alternatif de la figure des probabilités : ce qu'elle montre, en une phrase."""
    row = probabilities.reshape(DIGIT_COUNT)
    ranked = np.argsort(row)[::-1][:TOP_DIGITS_DESCRIBED]
    parts = ", ".join(f"{digit} à {row[digit]:.0%}" for digit in ranked)
    return f"Probabilités par chiffre ; les plus probables : {parts}."


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
        f"n°{neuron} {vote:+.1f}"
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
