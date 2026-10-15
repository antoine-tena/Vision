"""Un jeu d'images de chiffres : une ligne de 784 pixels (0 à 1) par image, et son chiffre."""

from dataclasses import dataclass

import numpy as np

from network.arrays import Labels, Matrix


@dataclass(frozen=True)
class DigitDataset:
    images: Matrix
    labels: Labels

    def __len__(self) -> int:
        """Nombre d'images du jeu."""
        return len(self.labels)

    def as_tuple(self) -> tuple[Matrix, Labels]:
        """(images, chiffres), la forme qu'attendent `train` et `accuracy`."""
        return self.images, self.labels

    @staticmethod
    def concatenate(*datasets: DigitDataset) -> DigitDataset:
        """Met plusieurs jeux bout à bout, par exemple MNIST et les chiffres imprimés."""
        return DigitDataset(
            np.concatenate([dataset.images for dataset in datasets]),
            np.concatenate([dataset.labels for dataset in datasets]),
        )
