"""Chargements mis en cache ([CACHE]), partagés par les pages : le modèle et ses rapports."""

import streamlit as st
from network.arrays import Matrix
from network.model import DEFAULT_MODEL_PATH, DigitClassifier, load_model
from network.training import EpochMetrics
from pipeline.sources import EVALUATION_FILE, HISTORY_FILE, load_evaluation, load_training_history


# Le modèle se charge une fois, pas à chaque trait.
@st.cache_resource
def cached_model() -> DigitClassifier:
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data
def cached_history() -> list[EpochMetrics]:
    return load_training_history(DEFAULT_MODEL_PATH.with_name(HISTORY_FILE))


@st.cache_data
def cached_evaluation() -> dict[str, Matrix]:
    return load_evaluation(DEFAULT_MODEL_PATH.with_name(EVALUATION_FILE))
