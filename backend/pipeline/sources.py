"""MNIST : téléchargé une fois dans data/raw/ (hors du dépôt), lu depuis le format IDX.

Source : base MNIST de Yann LeCun, Corinna Cortes et Christopher J. C. Burges, licence
CC BY-SA 3.0, servie par le miroir public de Google (voir data/README.md, [SOURCES]).
"""

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np

from network.arrays import Labels, Matrix
from network.model import DIGIT_COUNT, PIXEL_COUNT
from network.training import EpochMetrics
from pipeline.dataset import DigitDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MNIST_DIR = DATA_DIR / "raw" / "mnist"
# Échantillon du jeu d'entraînement écrit par network.sample, commité ([DONNEES-LOURDES]).
SAMPLE_PATH = DATA_DIR / "sample" / "exemples.npz"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
IMAGES_MAGIC = 2051
LABELS_MAGIC = 2049
PIXEL_MAX = 255.0
DOWNLOAD_TIMEOUT_SECONDS = 60
# Rapports écrits à côté du modèle par network.train et network.evaluate.
HISTORY_FILE = "historique.json"
EVALUATION_FILE = "evaluation.json"
HANDWRITTEN_SOURCE = "manuscrits"
PRINTED_SOURCE = "imprimés"


class DatasetFormatError(ValueError):
    """Fichier IDX inattendu : le chargement échoue, pas l'entraînement."""


def read_idx_images(payload: bytes) -> Matrix:
    magic, count, rows, columns = np.frombuffer(payload[:16], dtype=">i4")
    if magic != IMAGES_MAGIC or rows * columns != PIXEL_COUNT:
        raise DatasetFormatError(f"images IDX inattendues (magique {magic}, {rows}x{columns})")
    pixels = np.frombuffer(payload[16:], dtype=np.uint8).reshape(count, PIXEL_COUNT)
    images: Matrix = pixels.astype(np.float64) / PIXEL_MAX
    return images


def read_idx_labels(payload: bytes) -> Labels:
    magic, count = np.frombuffer(payload[:8], dtype=">i4")
    if magic != LABELS_MAGIC:
        raise DatasetFormatError(f"étiquettes IDX inattendues (magique {magic})")
    labels: Labels = np.frombuffer(payload[8:], dtype=np.uint8)[:count].astype(np.int64)
    return labels


def download_mnist(target_dir: Path = MNIST_DIR) -> None:
    """Télécharge les quatre fichiers absents ; ne refait rien s'ils sont déjà là."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_names in MNIST_FILES.values():
        for file_name in file_names:
            target = target_dir / file_name
            if not target.exists():
                url = f"{MNIST_URL}/{file_name}"
                with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                    target.write_bytes(response.read())


def load_mnist(split: Literal["train", "test"], source_dir: Path = MNIST_DIR) -> DigitDataset:
    images_file, labels_file = MNIST_FILES[split]
    if not (source_dir / images_file).exists():
        download_mnist(source_dir)
    images = read_idx_images(gzip.decompress((source_dir / images_file).read_bytes()))
    labels = read_idx_labels(gzip.decompress((source_dir / labels_file).read_bytes()))
    if len(images) != len(labels):
        raise DatasetFormatError(f"{len(images)} images pour {len(labels)} étiquettes")
    return DigitDataset(images, labels)


def load_training_history(path: Path) -> list[EpochMetrics]:
    """L'historique d'entraînement, une mesure par époque, validé ([SCHEMA-DONNEES])."""
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise DatasetFormatError(f"{path.name} : liste d'époques attendue, non vide")
    try:
        history = [EpochMetrics(**entry) for entry in entries]
    except TypeError as error:
        raise DatasetFormatError(f"{path.name} : époque mal formée ({error})") from error
    if any(not 0 <= metrics.validation_accuracy <= 1 for metrics in history):
        raise DatasetFormatError(f"{path.name} : exactitude hors de [0, 1]")
    return history


def load_evaluation(path: Path) -> dict[str, Matrix]:
    """Matrices de confusion du jeu de test, une par source, chacune 10x10 et positive."""
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict) or not entries:
        raise DatasetFormatError(f"{path.name} : une matrice par source attendue")
    try:
        matrices = {source: np.asarray(rows, dtype=np.float64) for source, rows in entries.items()}
    except (TypeError, ValueError) as error:
        raise DatasetFormatError(f"{path.name} : matrice illisible ({error})") from error
    for source, matrix in matrices.items():
        if matrix.shape != (DIGIT_COUNT, DIGIT_COUNT) or (matrix < 0).any():
            raise DatasetFormatError(f"{path.name} : matrice « {source} » invalide")
    return matrices


def load_training_sample(path: Path = SAMPLE_PATH) -> dict[str, DigitDataset]:
    """L'échantillon d'entraînement, un jeu par source, pixels ramenés entre 0 et 1."""
    with np.load(path) as arrays:
        sources = [
            name.removesuffix("_images") for name in arrays.files if name.endswith("_images")
        ]
        if not sources:
            raise DatasetFormatError(f"{path.name} : aucune image")
        sample = {
            source: DigitDataset(
                arrays[f"{source}_images"].astype(np.float64) / PIXEL_MAX,
                arrays[f"{source}_labels"].astype(np.int64),
            )
            for source in sources
        }
    for source, dataset in sample.items():
        _check_sample(path, source, dataset)
    return sample


def _check_sample(path: Path, source: str, dataset: DigitDataset) -> None:
    images, labels = dataset.as_tuple()
    if images.ndim != 2 or images.shape[1] != PIXEL_COUNT or len(images) != len(labels):
        raise DatasetFormatError(f"{path.name} : images « {source} » de forme {images.shape}")
    if ((labels < 0) | (labels >= DIGIT_COUNT)).any():
        raise DatasetFormatError(f"{path.name} : chiffre hors de 0-9 dans « {source} »")
