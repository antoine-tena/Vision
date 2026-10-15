"""Le réseau entier dessiné neurone par neurone : 784 pixels, la couche cachée, 10 sorties.

Un neurone pas encore atteint par la propagation est grisé ; un neurone atteint prend la couleur
et la taille de sa sortie ([FIGURES-ACCESSIBLES] : la taille double la couleur, les sorties
portent leur probabilité écrite).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from network.arrays import Matrix
from network.model import IMAGE_SIZE
from pipeline.activity import NetworkActivity
from pipeline.figures.common import COLORBLIND_PALETTE

PIXEL_SPACING = 0.1
HIDDEN_COLUMNS = 8
HIDDEN_LEFT = 3.8
HIDDEN_SPACING = 0.22
OUTPUT_X = 7.0
LAYER_HEIGHT = (IMAGE_SIZE - 1) * PIXEL_SPACING
TITLE_Y = LAYER_HEIGHT + 0.25
FRAME_PADDING = 0.2
FRAME_FILL_ALPHA = 0.06
OUTPUT_LABEL_WIDTH = 1.0
INPUT_FRAME_COLOR = "#BBBBBB"
FRAME_COLORS = (INPUT_FRAME_COLOR, COLORBLIND_PALETTE[2], COLORBLIND_PALETTE[3])
FRAME_DASHES = ("dot", "solid", "solid")
INPUT_FORMULA = "x = pixels ∈ [0, 1]"
HIDDEN_FORMULA = "h = max(0, x·W₁ + b₁)"
OUTPUT_FORMULA = "p = softmax(h·W₂ + b₂)"
UNREACHED = {"color": "#BBBBBB", "opacity": 0.35, "size": 5}
# Éteint : bleu-gris discret ; allumé : du turquoise au jaune vif, lisible sur fond sombre.
ACTIVITY_COLORSCALE = [[0.0, "#33415C"], [0.15, "#21918C"], [0.6, "#5EC962"], [1.0, "#FDE725"]]
EDGE_OPACITY = 0.7
EDGE_WIDTH = 1.5
FIGURE_HEIGHT = 420
PIXEL_SIZE = 9
HIDDEN_SIZES = (6, 16)
OUTPUT_SIZES = (12, 34)

type Positions = tuple[Matrix, Matrix]


def translucent(hex_color: str, alpha: float) -> str:
    red, green, blue = hex_to_rgb(hex_color)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def input_positions() -> Positions:
    rows, columns = np.divmod(np.arange(IMAGE_SIZE * IMAGE_SIZE), IMAGE_SIZE)
    return columns * PIXEL_SPACING, (IMAGE_SIZE - 1 - rows) * PIXEL_SPACING


def hidden_positions(size: int) -> Positions:
    rows, columns = np.divmod(np.arange(size), HIDDEN_COLUMNS)
    row_gaps = max(1, -(-size // HIDDEN_COLUMNS) - 1)
    return HIDDEN_LEFT + columns * HIDDEN_SPACING, LAYER_HEIGHT - rows * LAYER_HEIGHT / row_gaps


def output_positions(size: int) -> Positions:
    return np.full(size, OUTPUT_X), LAYER_HEIGHT - np.arange(size) * LAYER_HEIGHT / (size - 1)


def scaled_sizes(values: Matrix, sizes: tuple[int, int]) -> Matrix:
    """Taille de marqueur proportionnelle à la valeur, rapportée au maximum de la couche."""
    peak = float(values.max()) if values.size and values.max() > 0 else 1.0
    scaled: Matrix = sizes[0] + (sizes[1] - sizes[0]) * values / peak
    return scaled


def layer_traces(
    positions: Positions, reached: int, values: Matrix, marker: dict[str, object]
) -> list[go.Scatter]:
    """Une couche en deux tracés : les `reached` premiers neurones colorés, le reste grisé."""
    xs, ys = positions
    symbol = marker.get("symbol", "circle")
    return [
        go.Scatter(
            x=xs[reached:],
            y=ys[reached:],
            mode="markers",
            hoverinfo="skip",
            marker={**UNREACHED, "symbol": symbol},
        ),
        go.Scatter(
            x=xs[:reached],
            y=ys[:reached],
            mode="markers",
            customdata=values[:reached],
            hovertemplate="sortie %{customdata:.3f}<extra></extra>",
            marker={**marker, "color": values[:reached]},
        ),
    ]


def vote_edges(activity: NetworkActivity) -> list[go.Scatter]:
    """Liens des neurones cachés allumés vers le chiffre prédit : bleu pour, orange contre."""
    hidden_xs, hidden_ys = hidden_positions(len(activity.hidden))
    output_xs, output_ys = output_positions(len(activity.probabilities))
    traces = []
    for is_selected, color in (
        (activity.votes > 0, COLORBLIND_PALETTE[4]),
        (activity.votes < 0, COLORBLIND_PALETTE[1]),
    ):
        ends = np.ones(int(is_selected.sum()))
        xs = np.column_stack(
            [hidden_xs[is_selected], ends * output_xs[activity.digit], ends * np.nan]
        )
        ys = np.column_stack(
            [hidden_ys[is_selected], ends * output_ys[activity.digit], ends * np.nan]
        )
        traces.append(
            go.Scatter(
                x=xs.ravel(),
                y=ys.ravel(),
                mode="lines",
                hoverinfo="skip",
                line={"color": color, "width": EDGE_WIDTH},
                opacity=EDGE_OPACITY,
            )
        )
    return traces


def network_figure(activity: NetworkActivity, reached: int) -> go.Figure:
    """Le réseau quand `reached` neurones (numérotés de l'entrée à la sortie) sont atteints."""
    input_size, hidden_size, output_size = activity.layer_sizes
    hidden_reached = int(np.clip(reached - input_size, 0, hidden_size))
    output_reached = int(np.clip(reached - input_size - hidden_size, 0, output_size))
    traces = vote_edges(activity) if output_reached == output_size else []
    traces += layer_traces(
        input_positions(),
        min(reached, input_size),
        activity.pixels,
        {"symbol": "square", "size": PIXEL_SIZE, "colorscale": "gray", "cmin": 0, "cmax": 1},
    )
    traces += layer_traces(
        hidden_positions(hidden_size),
        hidden_reached,
        activity.hidden,
        {
            "size": scaled_sizes(activity.hidden, HIDDEN_SIZES)[:hidden_reached],
            "colorscale": ACTIVITY_COLORSCALE,
            "cmin": 0,
        },
    )
    traces += layer_traces(
        output_positions(output_size),
        output_reached,
        activity.probabilities,
        {
            "size": scaled_sizes(activity.probabilities, OUTPUT_SIZES)[:output_reached],
            "colorscale": ACTIVITY_COLORSCALE,
            "cmin": 0,
            "cmax": 1,
        },
    )
    figure = go.Figure(traces)
    return _network_layout(figure, activity, output_reached)


def _layer_frames(figure: go.Figure, layer_sizes: tuple[int, int, int]) -> None:
    """Un cadre et un titre par bloc ; seules les deux couches qui calculent sont numérotées,
    l'entrée (les pixels, sans poids) a un cadre gris en pointillés."""
    input_size, hidden_size, output_size = layer_sizes
    hidden_right = HIDDEN_LEFT + (HIDDEN_COLUMNS - 1) * HIDDEN_SPACING
    frames = (
        (0.0, LAYER_HEIGHT, f"Entrée : {input_size} pixels", INPUT_FORMULA),
        (HIDDEN_LEFT, hidden_right, f"Couche 1, cachée : {hidden_size} neurones", HIDDEN_FORMULA),
        (
            OUTPUT_X,
            OUTPUT_X + OUTPUT_LABEL_WIDTH,
            f"Couche 2, sortie : {output_size} chiffres",
            OUTPUT_FORMULA,
        ),
    )
    for index, (left, right, title, formula) in enumerate(frames):
        color = FRAME_COLORS[index]
        figure.add_shape(
            type="rect",
            x0=left - FRAME_PADDING,
            x1=right + FRAME_PADDING,
            y0=-FRAME_PADDING,
            y1=LAYER_HEIGHT + FRAME_PADDING,
            line={"color": color, "width": 2, "dash": FRAME_DASHES[index]},
            fillcolor=translucent(color, FRAME_FILL_ALPHA),
            layer="below",
        )
        figure.add_annotation(
            x=(left + right) / 2,
            y=TITLE_Y,
            yanchor="bottom",
            text=f"<b>{title}</b><br>{formula}",
            font={"color": color},
            showarrow=False,
        )


def _network_layout(figure: go.Figure, activity: NetworkActivity, output_reached: int) -> go.Figure:
    output_size = activity.layer_sizes[-1]
    output_xs, output_ys = output_positions(output_size)
    for digit in range(output_reached):
        figure.add_annotation(
            x=output_xs[digit] + 0.3,
            y=output_ys[digit],
            xanchor="left",
            showarrow=False,
            text=f"{digit} · {activity.probabilities[digit]:.0%}",
        )
    _layer_frames(figure, activity.layer_sizes)
    figure.update_layout(
        title="Le réseau en direct",
        height=FIGURE_HEIGHT,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(visible=False, range=[-0.4, OUTPUT_X + 1.4])
    figure.update_yaxes(visible=False, range=[-0.3, TITLE_Y + 0.6])
    return figure


def network_description(activity: NetworkActivity) -> str:
    input_size, hidden_size, _ = activity.layer_sizes
    active = int((activity.hidden > 0).sum())
    return (
        f"Le réseau entier : {input_size} pixels d'entrée, {active} neurones cachés allumés sur "
        f"{hidden_size} (plus gros et plus clairs quand ils répondent fort), puis les 10 sorties ; "
        f"le {activity.digit} l'emporte à {activity.probabilities[activity.digit]:.0%}. Les traits "
        f"montrent les votes pour (bleu) ou contre (orange) le {activity.digit}."
    )
