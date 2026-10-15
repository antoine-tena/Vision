"""Mesure le modèle enregistré sur les jeux de test et enregistre une matrice de confusion par
source : `uv run python -m network.evaluate` (depuis backend/). `network.train` le fait aussi.

Jeux de test : MNIST (manuscrits, jamais vus à l'entraînement) et chiffres imprimés générés avec
une autre graine que ceux de l'entraînement ([REPRODUCTIBLE]).
"""

import json
from pathlib import Path

import structlog

from network.model import DEFAULT_MODEL_PATH, DigitClassifier, load_model
from pipeline.dataset import DigitDataset
from pipeline.printed import generate_printed_digits
from pipeline.sources import EVALUATION_FILE, HANDWRITTEN_SOURCE, PRINTED_SOURCE, load_mnist
from pipeline.transforms import confusion_matrix

PRINTED_TEST_PER_DIGIT = 100
TEST_SEED_OFFSET = 1
DEFAULT_SEED = 0

logger = structlog.get_logger(__name__)


def build_test_sets(seed: int) -> dict[str, DigitDataset]:
    printed = generate_printed_digits(PRINTED_TEST_PER_DIGIT, seed + TEST_SEED_OFFSET)
    return {HANDWRITTEN_SOURCE: load_mnist("test"), PRINTED_SOURCE: printed}


def write_evaluation(
    model: DigitClassifier, test_sets: dict[str, DigitDataset], model_path: Path
) -> None:
    """Une matrice de confusion par source, dans `evaluation.json` à côté du modèle."""
    matrices = {
        source: confusion_matrix(dataset.labels, model.predict(dataset.images)).astype(int).tolist()
        for source, dataset in test_sets.items()
    }
    path = model_path.with_name(EVALUATION_FILE)
    path.write_text(json.dumps(matrices, ensure_ascii=False) + "\n")
    logger.info("evaluation.saved", path=str(path))


def main() -> None:
    write_evaluation(
        load_model(DEFAULT_MODEL_PATH), build_test_sets(DEFAULT_SEED), DEFAULT_MODEL_PATH
    )


if __name__ == "__main__":
    main()
