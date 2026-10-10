"""MNIST : téléchargé une fois dans data/raw/ (hors du dépôt), lu depuis le format IDX.

Source : base MNIST de Yann LeCun, Corinna Cortes et Christopher J. C. Burges, licence
CC BY-SA 3.0, servie par le miroir public de Google (voir data/README.md, [SOURCES]).
"""

import gzip
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np

from network.arrays import Labels, Matrix
from network.model import PIXEL_COUNT
from pipeline.dataset import DigitDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MNIST_DIR = DATA_DIR / "raw" / "mnist"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
IMAGES_MAGIC = 2051
LABELS_MAGIC = 2049
PIXEL_MAX = 255.0
DOWNLOAD_TIMEOUT_SECONDS = 60


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
