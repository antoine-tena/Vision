"""Page « Entraînement » : courbes d'apprentissage, mesure sur le jeu de test, poids appris."""

import streamlit as st
from network.arrays import Matrix
from network.model import DigitClassifier
from network.training import EpochMetrics
from pipeline.figures import (
    confusion_description,
    confusion_matrix_figure,
    hidden_patterns_figure,
    training_history_description,
    training_history_figure,
)
from pipeline.reasoning import hidden_patterns_mosaic
from pipeline.sources import DatasetFormatError
from pipeline.transforms import confusion_accuracy

from dashboard.constants import (
    MISSING_REPORT,
    PATTERN_GRID_COLUMNS,
    TRAINING_DESCRIPTION,
    TRAINING_TITLE,
)
from dashboard.loaders import cached_evaluation, cached_history, cached_model


def render_training() -> None:
    st.title(TRAINING_TITLE)
    st.write(TRAINING_DESCRIPTION)
    try:
        with st.spinner("Lecture du rapport d'entraînement…"):
            history, evaluation = cached_history(), cached_evaluation()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_REPORT.format(error=error))
        return
    render_history(history)
    render_evaluation(evaluation)
    render_hidden_patterns(cached_model())


def render_history(history: list[EpochMetrics]) -> None:
    st.subheader("Courbes d'apprentissage")
    last = history[-1]
    loss, train_accuracy, validation_accuracy = st.columns(3)
    loss.metric("Perte finale", f"{last.loss:.3f}")
    train_accuracy.metric("Exactitude (entraînement)", f"{last.train_accuracy:.1%}")
    validation_accuracy.metric("Exactitude (validation)", f"{last.validation_accuracy:.1%}")
    st.plotly_chart(training_history_figure(history))
    st.caption(training_history_description(history))


def render_evaluation(evaluation: dict[str, Matrix]) -> None:
    st.subheader("Jeu de test (images jamais vues)")
    source = st.radio("Source des images", list(evaluation), horizontal=True)
    matrix = evaluation[source]
    st.metric(f"Exactitude sur les {source}", f"{confusion_accuracy(matrix):.1%}")
    st.plotly_chart(confusion_matrix_figure(matrix))
    st.caption(confusion_description(matrix))


def render_hidden_patterns(model: DigitClassifier) -> None:
    st.subheader("Ce que la couche cachée a appris")
    st.plotly_chart(hidden_patterns_figure(hidden_patterns_mosaic(model, PATTERN_GRID_COLUMNS)))
    st.caption(
        f"Les {model.hidden.weights.shape[1]} neurones cachés, chacun en 28 x 28 : en bleu les "
        "pixels où il attend de l'encre, en orange ceux où l'encre le fait taire."
    )
