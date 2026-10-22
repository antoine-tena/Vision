"""Figures d'une reconnaissance : probabilités, pixels décisifs, neurones influents."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from network.arrays import Matrix
from network.model import DIGIT_COUNT
from pipeline.figures.common import (
    DIGITS,
    DIVERGING_COLORSCALE,
)
from pipeline.reasoning import Reasoning

PREDICTED_PATTERN = "/"
TOP_DIGITS_DESCRIBED = 3
NEURON_GRID_COLUMNS = 4
# Couche 2 (les sorties) en rose, comme dans la présentation ; scores bruts en jaune.
OUTPUT_COLOR = "#CC79A7"
MUTED_OUTPUT_COLOR = "#6E5A68"
SCORE_COLOR = "#E9C46A"
# Signe d'un poids, comme les traits du réseau : bleu pour, orange contre.
SUPPORT_COLOR = "#56B4E9"
AGAINST_COLOR = "#E69F00"
SHAPE_TITLE_SIZE = 11


def probabilities_figure(probabilities: Matrix) -> go.Figure:
    """Probabilité donnée à chaque chiffre pour une image ; la prédiction ressort par sa
    couleur, un motif hachuré et son pourcentage écrit ([ACCESSIBLE-FIGURES])."""
    row = probabilities.reshape(DIGIT_COUNT)
    predicted = int(np.argmax(row))
    is_predicted = [digit == predicted for digit in range(DIGIT_COUNT)]
    figure = go.Figure(
        go.Bar(
            x=DIGITS,
            y=row * 100,
            marker={
                "color": [OUTPUT_COLOR if flag else MUTED_OUTPUT_COLOR for flag in is_predicted],
                "pattern": {"shape": [PREDICTED_PATTERN if flag else "" for flag in is_predicted]},
            },
            text=[f"{row[predicted]:.0%} (prédit)" if flag else "" for flag in is_predicted],
            textposition="outside",
        )
    )
    figure.update_layout(title="Probabilités", xaxis_title="Chiffre", yaxis_title="Probabilité (%)")
    figure.update_xaxes(type="category")
    return figure


def scores_figure(scores: Matrix) -> go.Figure:
    """Les 10 scores bruts de la sortie, avant la softmax : des nombres quelconques."""
    figure = go.Figure(
        go.Bar(
            x=DIGITS,
            y=scores,
            marker={"color": SCORE_COLOR},
            text=[f"{score:+.1f}" for score in scores],
            textposition="outside",
        )
    )
    figure.update_layout(title="Scores bruts", xaxis_title="Chiffre", yaxis_title="Score z")
    figure.update_xaxes(type="category")
    return figure


def output_weights_figure(weights: Matrix, neuron: int) -> go.Figure:
    """Les 10 poids de sortie d'un neurone caché, un par chiffre : bleu si son motif pousse
    vers ce chiffre, orange s'il l'en éloigne ; la valeur est écrite ([ACCESSIBLE-FIGURES])."""
    figure = go.Figure(
        go.Bar(
            x=DIGITS,
            y=weights,
            marker={
                "color": [SUPPORT_COLOR if weight > 0 else AGAINST_COLOR for weight in weights]
            },
            text=[f"{weight:+.1f}" for weight in weights],
            textposition="outside",
        )
    )
    figure.update_layout(
        title=f"Ses 10 poids de sortie (neurone n°{neuron})",
        xaxis_title="Vers le chiffre",
        yaxis_title="Poids",
    )
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
    titles = [
        f"n°{neuron} {vote:+.1f}<br>{shape[0]}<br>{shape[1]}"
        for neuron, vote, shape in zip(
            reasoning.neuron_ids, reasoning.neuron_votes, reasoning.neuron_shapes, strict=True
        )
    ]
    return patterns_figure(
        reasoning.neuron_patterns,
        titles,
        f"Neurones qui ont le plus voté pour ou contre le {reasoning.digit}",
    )


def patterns_figure(patterns: Matrix, titles: list[str], title: str) -> go.Figure:
    """Motifs 28x28 (poids d'un neurone) en grille, chacun carré sous son titre."""
    count = len(patterns)
    figure = make_subplots(
        rows=max(1, -(-count // NEURON_GRID_COLUMNS)),
        cols=NEURON_GRID_COLUMNS,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.28,
    )
    for index, pattern in enumerate(patterns):
        figure.add_trace(
            go.Heatmap(
                z=pattern, coloraxis="coloraxis", hovertemplate="poids %{z:+.3f}<extra></extra>"
            ),
            row=index // NEURON_GRID_COLUMNS + 1,
            col=index % NEURON_GRID_COLUMNS + 1,
        )
    limit = float(np.abs(patterns).max()) if count else 1.0
    figure.update_layout(
        title=title,
        coloraxis={
            "colorscale": DIVERGING_COLORSCALE,
            "cmin": -limit,
            "cmax": limit,
            "colorbar": {"title": "Poids"},
        },
    )
    figure.update_annotations(font_size=SHAPE_TITLE_SIZE)
    figure.update_xaxes(showticklabels=False, ticks="", constrain="domain")
    figure.update_yaxes(showticklabels=False, ticks="", autorange="reversed")
    # Chaque motif reste carré : son axe vertical suit l'échelle de son propre axe horizontal.
    for index in range(count):
        suffix = "" if index == 0 else str(index + 1)
        figure.layout[f"yaxis{suffix}"].scaleanchor = f"x{suffix}"
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
