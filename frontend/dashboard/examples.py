"""Onglet « Exemples d'entraînement » : des chiffres que le réseau a vus en apprenant."""

import streamlit as st
from network.arrays import Matrix
from network.model import DIGIT_COUNT
from pipeline.sources import DatasetFormatError

from dashboard.constants import MISSING_SAMPLE
from dashboard.loaders import cached_training_sample


def pick_example() -> Matrix | None:
    """Les 784 pixels de l'exemple choisi, ou None si l'échantillon manque."""
    try:
        with st.spinner("Lecture des exemples…"):
            sample = cached_training_sample()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_SAMPLE.format(error=error))
        return None
    source_column, digit_column, number_column = st.columns(3)
    source = source_column.radio("Source", list(sample), help="MNIST ou chiffres imprimés")
    digit = digit_column.selectbox("Chiffre", range(DIGIT_COUNT))
    examples = sample[source].of_digit(digit)
    if not len(examples):
        st.warning(f"Aucun exemple de {digit} parmi les {source}.")
        return None
    number = number_column.slider(
        "Exemple n°", 1, len(examples), help="Tous tirés du jeu d'entraînement"
    )
    return examples.images[number - 1 : number]
