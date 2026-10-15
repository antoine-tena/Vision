"""Entraînement par mini-lots mélangés, avec une mesure par époque."""

from dataclasses import dataclass

import numpy as np

from network.arrays import Labels, Matrix
from network.model import DigitClassifier
from network.optimizers import StochasticGradientDescent


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparamètres de l'entraînement ; la graine rend le résultat reproductible."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.1
    decay: float = 1e-4
    momentum: float = 0.9
    seed: int = 0


@dataclass(frozen=True)
class EpochMetrics:
    """Mesures prises à la fin d'une époque (un passage complet sur les données)."""

    epoch: int
    loss: float
    train_accuracy: float
    validation_accuracy: float


def accuracy(model: DigitClassifier, images: Matrix, labels: Labels) -> float:
    """Part des images dont le chiffre prédit est le bon, entre 0 et 1."""
    return float(np.mean(model.predict(images) == labels))


def train(
    model: DigitClassifier,
    training: tuple[Matrix, Labels],
    validation: tuple[Matrix, Labels],
    config: TrainingConfig,
) -> list[EpochMetrics]:
    """Entraîne le modèle en place et renvoie l'historique, une mesure par époque.

    Après chaque époque, l'exactitude est mesurée sur l'entraînement et sur la validation,
    des images jamais apprises : un écart grandissant signale le surapprentissage.
    """
    optimizer = StochasticGradientDescent(config.learning_rate, config.decay, config.momentum)
    rng = np.random.default_rng(config.seed)
    history = []
    for epoch in range(1, config.epochs + 1):
        loss = _train_epoch(model, training, optimizer, config.batch_size, rng)
        history.append(
            EpochMetrics(epoch, loss, accuracy(model, *training), accuracy(model, *validation))
        )
    return history


def _train_epoch(
    model: DigitClassifier,
    training: tuple[Matrix, Labels],
    optimizer: StochasticGradientDescent,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """Une passe sur les données dans un ordre aléatoire ; renvoie la perte moyenne.

    Les images sont mélangées puis découpées en mini-lots d'environ `batch_size` ; chaque lot
    fait un pas d'apprentissage.
    """
    images, labels = training
    order = rng.permutation(len(labels))
    losses = [
        model.train_step(images[batch], labels[batch], optimizer)
        for batch in np.array_split(order, max(1, len(order) // batch_size))
    ]
    return float(np.mean(losses))
