from pathlib import Path

import numpy as np
import pytest

from network.model import PIXEL_COUNT, DigitClassifier
from pipeline.activity import network_activity, propagation_schedule
from pipeline.dataset import DigitDataset
from pipeline.figures import network_description, network_figure
from pipeline.recognition import recognize_pixels
from pipeline.sources import DatasetFormatError, load_training_sample
from pipeline.transforms import sample_per_digit, split_indices


def some_pixels() -> np.ndarray:
    return np.random.default_rng(0).random((1, PIXEL_COUNT))


def test_activity_follows_the_model() -> None:
    model = DigitClassifier(hidden_size=12, seed=1)

    activity = network_activity(model, some_pixels())

    assert activity.layer_sizes == (PIXEL_COUNT, 12, 10)
    assert activity.neuron_count == PIXEL_COUNT + 22
    assert np.allclose(activity.probabilities, model.predict_proba(some_pixels())[0])
    assert activity.digit == recognize_pixels(model, some_pixels()).digit
    assert (activity.hidden >= 0).all()


def test_schedule_fills_each_layer_in_order_and_ends_full() -> None:
    schedule = propagation_schedule((784, 128, 10), (4, 8, 1))

    assert len(schedule) == 13
    assert (np.diff(schedule) > 0).all()
    assert schedule[3] == 784 and schedule[11] == 912 and schedule[-1] == 922


@pytest.mark.parametrize("reached", [0, 300, 784 + 6, 784 + 12 + 10])
def test_network_figure_draws_every_neuron_at_every_step(reached: int) -> None:
    activity = network_activity(DigitClassifier(hidden_size=12), some_pixels())

    figure = network_figure(activity, reached)

    markers = [trace for trace in figure.data if trace.mode == "markers"]
    assert sum(len(trace.x) for trace in markers) == activity.neuron_count
    has_edges = any(trace.mode == "lines" for trace in figure.data)
    assert has_edges == (reached == activity.neuron_count)


def test_network_description_names_the_winner() -> None:
    activity = network_activity(DigitClassifier(hidden_size=12), some_pixels())

    assert f"le {activity.digit} l'emporte" in network_description(activity)


def test_split_indices_cover_everything_once() -> None:
    training, validation = split_indices(50, 0.2, seed=3)

    assert len(validation) == 10
    assert sorted(np.concatenate([training, validation]).tolist()) == list(range(50))


def test_sample_takes_at_most_n_of_each_digit() -> None:
    labels = np.array([0] * 5 + [1] * 2 + [2] * 4)
    dataset = DigitDataset(np.arange(len(labels))[:, None].astype(float), labels)

    sample = sample_per_digit(dataset, 3, seed=0)

    assert sample.labels.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert len(sample.of_digit(1)) == 2


def test_training_sample_is_loaded_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "exemples.npz"
    np.savez(path, manuscrits_images=np.full((2, PIXEL_COUNT), 255, np.uint8),
             manuscrits_labels=np.array([3, 4], np.uint8))  # fmt: skip

    sample = load_training_sample(path)

    assert sample["manuscrits"].images.max() == 1.0
    assert sample["manuscrits"].labels.tolist() == [3, 4]

    np.savez(path, manuscrits_images=np.zeros((2, 10), np.uint8),
             manuscrits_labels=np.array([3, 4], np.uint8))  # fmt: skip
    with pytest.raises(DatasetFormatError):
        load_training_sample(path)


def test_committed_sample_has_every_digit() -> None:
    for dataset in load_training_sample().values():
        assert set(dataset.labels.tolist()) == set(range(10))
