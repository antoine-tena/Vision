"""Un jeu d'images de chiffres : une ligne de 784 pixels (0 à 1) par image, et son chiffre."""

from dataclasses import dataclass

import numpy as np

from network.arrays import Labels, Matrix


@dataclass(frozen=True)
class DigitDataset:
    images: Matrix
    labels: Labels

    def __len__(self) -> int:
        return len(self.labels)

    def as_tuple(self) -> tuple[Matrix, Labels]:
        return self.images, self.labels

    @staticmethod
    def concatenate(*datasets: DigitDataset) -> DigitDataset:
        return DigitDataset(
            np.concatenate([dataset.images for dataset in datasets]),
            np.concatenate([dataset.labels for dataset in datasets]),
        )
