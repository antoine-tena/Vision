"""Le classifieur de chiffres : 784 pixels → couche cachée (ReLU) → 10 probabilités."""

from pathlib import Path

import numpy as np

from network.activations import ReLU, softmax
from network.arrays import Labels, Matrix
from network.layers import DenseLayer
from network.losses import SoftmaxCrossEntropy
from network.optimizers import StochasticGradientDescent

IMAGE_SIZE = 28
PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE
DIGIT_COUNT = 10
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "vision.npz"


class DigitClassifier:
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = DenseLayer(PIXEL_COUNT, hidden_size, rng)
        self.activation = ReLU()
        self.output = DenseLayer(hidden_size, DIGIT_COUNT, rng)
        self.loss = SoftmaxCrossEntropy()

    @property
    def layers(self) -> list[DenseLayer]:
        return [self.hidden, self.output]

    def logits(self, images: Matrix) -> Matrix:
        return self.output.forward(self.activation.forward(self.hidden.forward(images)))

    def predict_proba(self, images: Matrix) -> Matrix:
        """Probabilité de chaque chiffre, une ligne par image (pixels entre 0 et 1)."""
        return softmax(self.logits(images))

    def predict(self, images: Matrix) -> Labels:
        predictions: Labels = np.argmax(self.logits(images), axis=1).astype(np.int64)
        return predictions

    def train_step(
        self, images: Matrix, labels: Labels, optimizer: StochasticGradientDescent
    ) -> float:
        """Propagation, perte, rétropropagation puis mise à jour des poids ; renvoie la perte."""
        loss = self.loss.forward(self.logits(images), labels)
        gradients = self.output.backward(self.loss.backward())
        self.hidden.backward(self.activation.backward(gradients))
        optimizer.step(self.layers)
        return loss

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            hidden_weights=self.hidden.weights.astype(np.float32),
            hidden_biases=self.hidden.biases.astype(np.float32),
            output_weights=self.output.weights.astype(np.float32),
            output_biases=self.output.biases.astype(np.float32),
        )


def load_model(path: Path = DEFAULT_MODEL_PATH) -> DigitClassifier:
    with np.load(path) as weights:
        return _with_weights(
            weights["hidden_weights"],
            weights["hidden_biases"],
            weights["output_weights"],
            weights["output_biases"],
        )


def _with_weights(
    hidden_weights: Matrix, hidden_biases: Matrix, output_weights: Matrix, output_biases: Matrix
) -> DigitClassifier:
    model = DigitClassifier(hidden_size=hidden_weights.shape[1])
    model.hidden.weights = hidden_weights.astype(np.float64)
    model.hidden.biases = hidden_biases.astype(np.float64)
    model.output.weights = output_weights.astype(np.float64)
    model.output.biases = output_biases.astype(np.float64)
    return model
