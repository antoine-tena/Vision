"""Extrait un échantillon du jeu d'entraînement, commité pour le tableau de bord :
`uv run python -m network.sample` (depuis backend/).

Les images viennent de la part « entraînement » du découpage de `network.train` (même graine),
donc le modèle les a vraiment vues ; pas une de la validation.
"""

from typing import Any

import numpy as np
import structlog

from network.train import PRINTED_PER_DIGIT, VALIDATION_FRACTION
from pipeline.dataset import DigitDataset
from pipeline.printed import generate_printed_digits
from pipeline.sources import (
    HANDWRITTEN_SOURCE,
    PIXEL_MAX,
    PRINTED_SOURCE,
    SAMPLE_PATH,
    load_mnist,
)
from pipeline.transforms import sample_per_digit, split_indices

SAMPLE_PER_DIGIT = 20
DEFAULT_SEED = 0

logger = structlog.get_logger(__name__)


def training_parts(seed: int) -> dict[str, DigitDataset]:
    """Les images d'entraînement de chaque source, sans celles mises de côté pour la validation."""
    handwritten = load_mnist("train")
    printed = generate_printed_digits(PRINTED_PER_DIGIT, seed)
    training, _ = split_indices(len(handwritten) + len(printed), VALIDATION_FRACTION, seed)
    is_handwritten = training < len(handwritten)
    return {
        HANDWRITTEN_SOURCE: handwritten.subset(training[is_handwritten]),
        PRINTED_SOURCE: printed.subset(training[~is_handwritten] - len(handwritten)),
    }


def main() -> None:
    # Any : les stubs de savez_compressed confondent **arrays et le mot-clé allow_pickle.
    arrays: dict[str, Any] = {}
    for source, dataset in training_parts(DEFAULT_SEED).items():
        sample = sample_per_digit(dataset, SAMPLE_PER_DIGIT, DEFAULT_SEED)
        arrays[f"{source}_images"] = np.round(sample.images * PIXEL_MAX).astype(np.uint8)
        arrays[f"{source}_labels"] = sample.labels.astype(np.uint8)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAMPLE_PATH, **arrays)
    logger.info("sample.saved", path=str(SAMPLE_PATH))


if __name__ == "__main__":
    main()
