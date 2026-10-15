"""Entraîne le modèle et l'enregistre : `uv run python -m network.train` (depuis backend/).

Données : MNIST (manuscrits) et chiffres imprimés générés ; graine fixée, donc le même modèle
d'un lancement à l'autre sur un même poste ([REPRODUCTIBLE]).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import structlog

from network.evaluate import build_test_sets, write_evaluation
from network.model import DEFAULT_MODEL_PATH, DigitClassifier
from network.training import TrainingConfig, accuracy, train
from pipeline.dataset import DigitDataset
from pipeline.printed import generate_printed_digits
from pipeline.sources import HISTORY_FILE, load_mnist
from pipeline.transforms import split_dataset

VALIDATION_FRACTION = 0.1
PRINTED_PER_DIGIT = 600

logger = structlog.get_logger(__name__)


def build_datasets(seed: int) -> tuple[DigitDataset, DigitDataset]:
    """Entraînement et validation, chacun mêlant manuscrits et imprimés."""
    printed = generate_printed_digits(PRINTED_PER_DIGIT, seed)
    return split_dataset(
        DigitDataset.concatenate(load_mnist("train"), printed), VALIDATION_FRACTION, seed
    )


def run(config: TrainingConfig, model_path: Path) -> None:
    training, validation = build_datasets(config.seed)
    logger.info("training.started", examples=len(training), **asdict(config))
    model = DigitClassifier(seed=config.seed)
    history = train(model, training.as_tuple(), validation.as_tuple(), config)
    for metrics in history:
        logger.info("training.epoch", **asdict(metrics))
    model.save(model_path)
    model_path.with_name(HISTORY_FILE).write_text(
        json.dumps([asdict(metrics) for metrics in history], indent=2) + "\n"
    )
    test_sets = build_test_sets(config.seed)
    test = DigitDataset.concatenate(*test_sets.values())
    logger.info(
        "training.saved", path=str(model_path), test_accuracy=accuracy(model, *test.as_tuple())
    )
    write_evaluation(model, test_sets, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    arguments = parser.parse_args()
    config = TrainingConfig(
        epochs=arguments.epochs, learning_rate=arguments.learning_rate, seed=arguments.seed
    )
    run(config, arguments.output)


if __name__ == "__main__":
    main()
