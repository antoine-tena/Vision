"""Nom lisible du motif guetté par un neurone caché : forme (trait, boucle, tache) et zone.

Lecture approchée de ses poids : on ne garde que les plus forts poids positifs (là où le neurone
veut de l'encre), puis on regarde où ils sont (centre de gravité) et comment ils s'étalent
(allongés dans un sens : un trait ; en anneau autour d'un creux : une boucle ; sinon une tache).
"""

from collections import Counter

import numpy as np

from network.arrays import Matrix

STRONG_WEIGHT_SHARE = 0.6
ELONGATION_RATIO = 2.5
HOLLOW_CENTER_SHARE = 0.08
HOLLOW_RADIUS_SHARE = 0.5
# Le chiffre occupe le centre de l'image : la zone du milieu est une bande étroite.
UPPER_LIMIT = 11.5
LOWER_LIMIT = 15.5
DIAGONAL_TOLERANCE_DEGREES = 25
ROWS = ("en haut", "au milieu", "en bas")
COLUMNS = ("à gauche", "", "à droite")


def zone_name(row: float, column: float) -> str:
    vertical = ROWS[int(row > UPPER_LIMIT) + int(row > LOWER_LIMIT)]
    horizontal = COLUMNS[int(column > UPPER_LIMIT) + int(column > LOWER_LIMIT)]
    if vertical == ROWS[1] and not horizontal:
        return "au centre"
    return f"{vertical} {horizontal}".strip()


def stroke_name(direction: Matrix) -> str:
    """Nom du trait d'après sa direction (ligne, colonne), les lignes allant vers le bas."""
    angle = float(np.degrees(np.arctan2(-direction[0], direction[1]))) % 180
    if angle < DIAGONAL_TOLERANCE_DEGREES or angle > 180 - DIAGONAL_TOLERANCE_DEGREES:
        return "trait horizontal"
    if abs(angle - 90) < DIAGONAL_TOLERANCE_DEGREES:
        return "trait vertical"
    return "diagonale /" if angle < 90 else "diagonale \\"


def shape_parts(pattern: Matrix) -> tuple[str, str]:
    """Forme et zone du motif d'un neurone (ses 784 poids en 28 x 28) : (« boucle », « en bas »)."""
    ink = np.where(pattern >= STRONG_WEIGHT_SHARE * pattern.max(), pattern, 0.0)
    rows, columns = np.indices(ink.shape)
    mass = float(ink.sum())
    if mass <= 0:
        return "aucun motif net", ""
    center = np.array([(rows * ink).sum(), (columns * ink).sum()]) / mass
    offsets = np.stack([rows - center[0], columns - center[1]], axis=-1)
    covariance = np.einsum("ijk,ijl,ij->kl", offsets, offsets, ink) / mass
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    zone = zone_name(float(center[0]), float(center[1]))
    if eigenvalues[1] > ELONGATION_RATIO * max(float(eigenvalues[0]), 1e-9):
        return stroke_name(eigenvectors[:, 1]), zone
    distances = np.hypot(offsets[..., 0], offsets[..., 1])
    radius = float((distances * ink).sum()) / mass
    is_hollow = ink[distances < HOLLOW_RADIUS_SHARE * radius].sum() < HOLLOW_CENTER_SHARE * mass
    return ("boucle" if is_hollow else "tache"), zone


def shape_name(pattern: Matrix) -> str:
    """Motif d'un neurone en quelques mots : « boucle en bas »."""
    return " ".join(part for part in shape_parts(pattern) if part)


def active_shapes(patterns: Matrix, activations: Matrix) -> list[tuple[str, int, int]]:
    """Motifs des neurones allumés, regroupés, les plus fréquents d'abord : (motif, nombre,
    neurone le plus allumé de ce motif, qui sert d'exemple)."""
    names = {
        int(neuron): shape_name(patterns[neuron]) for neuron in np.flatnonzero(activations > 0)
    }
    counts = Counter(names.values())
    examples: dict[str, int] = {}
    for neuron in sorted(names, key=lambda neuron: -activations[neuron]):
        examples.setdefault(names[neuron], neuron)
    return [(name, count, examples[name]) for name, count in counts.most_common()]
