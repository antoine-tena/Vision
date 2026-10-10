import gzip
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from network.model import IMAGE_SIZE, PIXEL_COUNT
from network.training import EpochMetrics
from pipeline.dataset import DigitDataset
from pipeline.figures import (
    confusion_matrix_figure,
    probabilities_description,
    probabilities_figure,
    training_history_figure,
)
from pipeline.printed import generate_printed_digits
from pipeline.sources import DatasetFormatError, load_mnist, read_idx_images, read_idx_labels
from pipeline.transforms import confusion_matrix, digit_image, split_dataset, to_mnist_pixels


def idx_images(count: int) -> bytes:
    header = np.array([2051, count, IMAGE_SIZE, IMAGE_SIZE], dtype=">i4").tobytes()
    return header + np.full(count * PIXEL_COUNT, 255, dtype=np.uint8).tobytes()


def idx_labels(labels: list[int]) -> bytes:
    return np.array([2049, len(labels)], dtype=">i4").tobytes() + bytes(labels)


def test_idx_files_are_read_into_scaled_pixels_and_labels() -> None:
    assert read_idx_images(idx_images(2)).shape == (2, PIXEL_COUNT)
    assert read_idx_images(idx_images(1)).max() == 1.0
    assert read_idx_labels(idx_labels([3, 7])).tolist() == [3, 7]


def test_unexpected_idx_file_fails_at_load() -> None:
    with pytest.raises(DatasetFormatError):
        read_idx_labels(idx_images(1))


def test_mnist_is_read_from_local_files(tmp_path: Path) -> None:
    (tmp_path / "t10k-images-idx3-ubyte.gz").write_bytes(gzip.compress(idx_images(2)))
    (tmp_path / "t10k-labels-idx1-ubyte.gz").write_bytes(gzip.compress(idx_labels([1, 2])))

    dataset = load_mnist("test", source_dir=tmp_path)

    assert len(dataset) == 2


def dark_digit_on_white(offset: int) -> Image.Image:
    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).rectangle((offset, 20, offset + 10, 80), fill=0)
    return image


def test_drawn_digit_is_inverted_and_centered_like_mnist() -> None:
    pixels = to_mnist_pixels(dark_digit_on_white(offset=5)).reshape(IMAGE_SIZE, IMAGE_SIZE)

    rows, columns = np.indices(pixels.shape)
    assert pixels.max() > 0.9
    assert abs((rows * pixels).sum() / pixels.sum() - IMAGE_SIZE / 2) < 1.5
    assert abs((columns * pixels).sum() / pixels.sum() - IMAGE_SIZE / 2) < 1.5


def test_blank_image_gives_blank_pixels() -> None:
    assert to_mnist_pixels(Image.new("L", (50, 50), 255)).sum() == 0


def test_split_is_reproducible() -> None:
    dataset = DigitDataset(np.zeros((100, PIXEL_COUNT)), np.arange(100) % 10)

    training, validation = split_dataset(dataset, 0.2, seed=1)
    again, _ = split_dataset(dataset, 0.2, seed=1)

    assert (len(training), len(validation)) == (80, 20)
    assert np.array_equal(training.labels, again.labels)


def test_printed_digits_work_with_the_bundled_font() -> None:
    printed = generate_printed_digits(2, seed=0, fonts=[])

    assert printed.images.shape == (20, PIXEL_COUNT)
    assert printed.labels.tolist() == [digit for digit in range(10) for _ in range(2)]
    assert (printed.images.sum(axis=1) > 0).all()


def test_confusion_matrix_counts_each_pair() -> None:
    matrix = confusion_matrix(np.array([1, 1, 2]), np.array([1, 2, 2]))

    assert matrix[1, 1] == 1 and matrix[1, 2] == 1 and matrix[2, 2] == 1
    assert len(confusion_matrix_figure(matrix).data) == 1


def test_figures_have_titled_axes() -> None:
    history = [EpochMetrics(1, 0.5, 0.8, 0.75), EpochMetrics(2, 0.3, 0.9, 0.85)]
    probabilities = np.full((1, 10), 0.1)

    assert training_history_figure(history).layout.xaxis.title.text == "Époque"
    assert probabilities_figure(probabilities).layout.yaxis.title.text == "Probabilité (%)"


def test_recognition_returns_the_most_probable_digit() -> None:
    from network.model import DigitClassifier
    from pipeline.recognition import recognize

    recognition = recognize(DigitClassifier(hidden_size=8), dark_digit_on_white(offset=40))

    assert recognition.digit == int(recognition.probabilities.argmax())
    assert np.isclose(recognition.probabilities.sum(), 1.0)
    assert not recognition.is_blank
    assert recognize(DigitClassifier(hidden_size=8), Image.new("L", (40, 40), 255)).is_blank


def test_digit_image_is_a_bounded_square() -> None:
    image = digit_image(np.linspace(-1.0, 2.0, PIXEL_COUNT))

    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert image.min() == 0.0 and image.max() == 1.0


def test_predicted_digit_stands_out_without_color() -> None:
    probabilities = np.full((1, 10), 0.05)
    probabilities[0, 7] = 0.55

    bar = probabilities_figure(probabilities).data[0]

    assert [shape for shape in bar.marker.pattern.shape if shape] == ["/"]
    assert bar.text[7] == "55% (prédit)"
    assert probabilities_description(probabilities).startswith(
        "Probabilités par chiffre ; les plus probables : 7 à 55%"
    )
