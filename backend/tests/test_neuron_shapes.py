import numpy as np

from network.model import IMAGE_SIZE
from pipeline.neuron_shapes import active_shapes, shape_name


def blank() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE))


def test_horizontal_stroke_at_top() -> None:
    pattern = blank()
    pattern[6, 6:22] = 1.0
    assert shape_name(pattern) == "trait horizontal en haut"


def test_vertical_stroke_at_center() -> None:
    pattern = blank()
    pattern[6:22, 13] = 1.0
    assert shape_name(pattern) == "trait vertical au centre"


def test_rising_diagonal() -> None:
    pattern = blank()
    for step in range(16):
        pattern[21 - step, 6 + step] = 1.0
    assert shape_name(pattern).startswith("diagonale /")


def test_loop_at_bottom() -> None:
    pattern = blank()
    rows, columns = np.indices(pattern.shape)
    pattern[np.abs(np.hypot(rows - 19, columns - 13) - 5) < 1] = 1.0
    assert shape_name(pattern) == "boucle en bas"


def test_no_positive_weight() -> None:
    assert shape_name(-np.ones((IMAGE_SIZE, IMAGE_SIZE))) == "aucun motif net"


def test_active_shapes_counts_only_lit_neurons() -> None:
    top, bottom = blank(), blank()
    top[6, 6:22] = 1.0
    bottom[21, 6:22] = 1.0
    shapes = active_shapes(np.stack([top, top, bottom]), np.array([1.0, 2.0, 0.0]))
    assert shapes == [("trait horizontal en haut", 2, 1)]
