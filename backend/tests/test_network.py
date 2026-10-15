from pathlib import Path

import numpy as np

from network.activations import softmax
from network.losses import cross_entropy
from network.model import PIXEL_COUNT, DigitClassifier, load_model
from network.optimizers import StochasticGradientDescent
from network.training import TrainingConfig, accuracy, train

FINITE_DIFFERENCE_STEP = 1e-5


def small_batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((6, PIXEL_COUNT)), rng.integers(0, 10, size=6).astype(np.int64)


def test_softmax_rows_are_probabilities() -> None:
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[1], 1 / 3)


def test_cross_entropy_of_a_sure_answer_is_small() -> None:
    assert cross_entropy(np.array([[0.0, 1.0]]), np.array([1])) < 1e-6
    assert np.isclose(cross_entropy(np.array([[0.5, 0.5]]), np.array([0])), np.log(2))


def test_backpropagation_matches_numerical_gradient() -> None:
    model = DigitClassifier(hidden_size=5, seed=1)
    images, labels = small_batch()
    model.loss.forward(model.logits(images), labels)
    model.hidden.backward(model.activation.backward(model.output.backward(model.loss.backward())))
    for row, column in [(0, 0), (100, 3), (783, 4)]:
        original = model.hidden.weights[row, column]
        model.hidden.weights[row, column] = original + FINITE_DIFFERENCE_STEP
        loss_up = model.loss.forward(model.logits(images), labels)
        model.hidden.weights[row, column] = original - FINITE_DIFFERENCE_STEP
        loss_down = model.loss.forward(model.logits(images), labels)
        model.hidden.weights[row, column] = original
        numerical = (loss_up - loss_down) / (2 * FINITE_DIFFERENCE_STEP)
        assert np.isclose(model.hidden.weight_gradients[row, column], numerical, atol=1e-6)


def test_a_training_step_lowers_the_loss() -> None:
    model = DigitClassifier(hidden_size=16, seed=2)
    images, labels = small_batch()
    optimizer = StochasticGradientDescent(learning_rate=0.1, momentum=0.9)

    first_loss = model.train_step(images, labels, optimizer)
    for _ in range(20):
        last_loss = model.train_step(images, labels, optimizer)

    assert last_loss < first_loss
    assert optimizer.iterations == 21


def test_training_learns_a_separable_toy_problem() -> None:
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 10, size=500).astype(np.int64)
    images = np.zeros((500, PIXEL_COUNT))
    images[np.arange(500), labels * 50] = 1.0  # un pixel allumé par chiffre
    model = DigitClassifier(hidden_size=32, seed=3)

    history = train(model, (images, labels), (images, labels), TrainingConfig(epochs=30, seed=3))

    assert history[-1].validation_accuracy == 1.0
    assert accuracy(model, images, labels) == 1.0


def test_saved_model_predicts_the_same(tmp_path: Path) -> None:
    model = DigitClassifier(hidden_size=8, seed=4)
    images, _ = small_batch()
    model.save(tmp_path / "modele.npz")

    restored = load_model(tmp_path / "modele.npz")

    assert np.array_equal(restored.predict(images), model.predict(images))


def test_parameter_count_adds_weights_and_biases() -> None:
    from network.model import DigitClassifier

    assert DigitClassifier(hidden_size=128).parameter_count == 784 * 128 + 128 + 128 * 10 + 10
