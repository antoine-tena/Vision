import numpy as np

from pipeline.transforms import top_confusions


def test_top_confusions_skip_the_diagonal_and_zeros() -> None:
    matrix = np.array([[50, 3, 0], [1, 40, 0], [7, 0, 30]])

    assert top_confusions(matrix, 5) == [(2, 0, 7), (0, 1, 3), (1, 0, 1)]
