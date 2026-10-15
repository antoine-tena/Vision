import json
from pathlib import Path

import numpy as np
import pytest

from network.evaluate import write_evaluation
from network.model import IMAGE_SIZE, PIXEL_COUNT, DigitClassifier
from network.training import EpochMetrics
from pipeline.dataset import DigitDataset
from pipeline.figures import (
    confusion_description,
    evidence_description,
    evidence_figure,
    hidden_patterns_figure,
    neurons_description,
    neurons_figure,
    training_history_description,
)
from pipeline.reasoning import explain, neuron_patterns, tile_patterns
from pipeline.sources import (
    EVALUATION_FILE,
    DatasetFormatError,
    load_evaluation,
    load_training_history,
)


def some_pixels(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((1, PIXEL_COUNT))


def test_score_is_the_logit_and_the_sum_of_the_votes() -> None:
    model = DigitClassifier(hidden_size=16, seed=2)
    pixels = some_pixels()

    reasoning = explain(model, pixels, digit=3, top_neurons=16)

    assert np.isclose(reasoning.score, model.logits(pixels)[0, 3])
    assert np.isclose(reasoning.neuron_votes.sum(), reasoning.score)


def test_evidence_adds_up_to_the_score_when_biases_are_zero() -> None:
    model = DigitClassifier(hidden_size=16, seed=3)
    pixels = some_pixels(1)

    reasoning = explain(model, pixels, digit=7)

    assert reasoning.evidence.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert np.isclose(reasoning.evidence.sum(), reasoning.score)


def test_neurons_are_ranked_by_influence() -> None:
    reasoning = explain(DigitClassifier(hidden_size=16, seed=4), some_pixels(), digit=0)

    influence = np.abs(reasoning.neuron_votes)
    assert len(reasoning.neuron_ids) == 8
    assert (np.diff(influence) <= 0).all()
    assert reasoning.neuron_patterns.shape == (8, IMAGE_SIZE, IMAGE_SIZE)
    assert 0 <= reasoning.active_count <= reasoning.hidden_size == 16


def test_blank_image_has_no_evidence() -> None:
    reasoning = explain(DigitClassifier(hidden_size=8), np.zeros((1, PIXEL_COUNT)), digit=1)

    assert not reasoning.evidence.any()


def test_patterns_are_tiled_with_blank_separators() -> None:
    patterns = neuron_patterns(DigitClassifier(hidden_size=5), np.arange(5))

    mosaic = tile_patterns(patterns, columns=2)

    assert mosaic.shape == (3 * IMAGE_SIZE + 2, 2 * IMAGE_SIZE + 1)
    assert np.isnan(mosaic[IMAGE_SIZE]).all()
    assert np.array_equal(mosaic[:IMAGE_SIZE, IMAGE_SIZE + 1 :], patterns[1])
    assert np.isnan(mosaic[-IMAGE_SIZE:, -IMAGE_SIZE:]).all()


def test_reasoning_figures_are_titled_and_described() -> None:
    reasoning = explain(DigitClassifier(hidden_size=16, seed=5), some_pixels(), digit=4)
    mosaic = tile_patterns(reasoning.neuron_patterns, columns=4)

    assert evidence_figure(reasoning.evidence, 4).layout.yaxis.title.text == "Ligne (pixel)"
    assert len(neurons_figure(reasoning).data) == 8
    assert len(hidden_patterns_figure(mosaic).data) == 1
    assert "le 4" in evidence_description(reasoning.evidence, 4)
    assert f"n°{reasoning.neuron_ids[0]}" in neurons_description(reasoning)
    assert "pour" in evidence_description(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), 4)


def test_report_descriptions_sum_up_the_figures() -> None:
    history = [EpochMetrics(1, 0.5, 0.8, 0.75), EpochMetrics(2, 0.3, 0.9, 0.85)]
    matrix = np.diag(np.full(10, 9.0))
    matrix[7, 1] = 10

    assert "2 époques" in training_history_description(history)
    assert "7 pris pour 1 (10 fois)" in confusion_description(matrix)


def test_training_history_is_loaded_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "historique.json"
    path.write_text(
        json.dumps([{"epoch": 1, "loss": 0.3, "train_accuracy": 0.9, "validation_accuracy": 0.8}])
    )
    assert load_training_history(path)[0].validation_accuracy == 0.8

    for broken in (
        [],
        [{"epoch": 1}],
        [{"epoch": 1, "loss": 0.3, "train_accuracy": 0.9, "validation_accuracy": 8}],
    ):
        path.write_text(json.dumps(broken))
        with pytest.raises(DatasetFormatError):
            load_training_history(path)


def test_evaluation_is_written_then_loaded(tmp_path: Path) -> None:
    labels = np.arange(10)
    test_sets = {"manuscrits": DigitDataset(np.zeros((10, PIXEL_COUNT)), labels)}

    write_evaluation(DigitClassifier(hidden_size=4), test_sets, tmp_path / "vision.npz")
    matrices = load_evaluation(tmp_path / EVALUATION_FILE)

    assert matrices["manuscrits"].shape == (10, 10)
    assert matrices["manuscrits"].sum() == 10


@pytest.mark.parametrize(
    "broken", [{}, {"manuscrits": [[1, 2], [3]]}, {"imprimés": [[-1] * 10] * 10}]
)
def test_broken_evaluation_is_refused(tmp_path: Path, broken: object) -> None:
    path = tmp_path / EVALUATION_FILE
    path.write_text(json.dumps(broken))

    with pytest.raises(DatasetFormatError):
        load_evaluation(path)


def test_evidence_totals_split_support_and_against() -> None:
    from pipeline.reasoning import evidence_totals

    assert evidence_totals(np.array([[1.0, -0.5], [0.0, 2.0]])) == (3.0, 0.5)


def test_before_and_after_alternates_initial_and_learned_patterns() -> None:
    from network.model import DigitClassifier
    from pipeline.reasoning import before_and_after, neuron_patterns

    model = DigitClassifier(hidden_size=4, seed=0)
    model.hidden.weights = model.hidden.weights + 1.0
    neurons = np.array([2, 0])

    patterns = before_and_after(model, neurons, seed=0)

    initial = neuron_patterns(DigitClassifier(hidden_size=4, seed=0), neurons)
    assert patterns.shape == (4, 28, 28)
    np.testing.assert_allclose(patterns[0], initial[0])
    np.testing.assert_allclose(patterns[1], initial[0] + 1.0)


def test_digit_scores_has_one_score_per_digit() -> None:
    from network.model import DigitClassifier
    from pipeline.reasoning import digit_scores

    assert digit_scores(DigitClassifier(hidden_size=4), np.zeros((1, 784))).shape == (10,)


def test_output_weights_are_the_neuron_row() -> None:
    from network.model import DigitClassifier
    from pipeline.reasoning import output_weights

    model = DigitClassifier(hidden_size=4)

    np.testing.assert_array_equal(output_weights(model, 2), model.output.weights[2])
