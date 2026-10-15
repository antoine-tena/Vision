"""Partie 2 du diaporama, l'entraînement : les données, les courbes, le test."""

import streamlit as st
from network.model import DigitClassifier
from network.split_sizes import split_sizes
from pipeline.figures import confusion_matrix_figure, training_history_figure
from pipeline.recognition import Recognition
from pipeline.sources import DatasetFormatError
from pipeline.transforms import confusion_accuracy, top_confusions

from dashboard import explanations
from dashboard.constants import MISSING_REPORT, SLIDE_FIGURE_HEIGHT, TOP_CONFUSIONS
from dashboard.explanations import key_fact
from dashboard.formula import Term, show_formula
from dashboard.loaders import cached_evaluation, cached_history

TEXT_SHARE = (2, 3)


def render_split_slide(model: DigitClassifier, recognition: Recognition) -> None:
    sizes = split_sizes()
    key_fact(f"{sizes.total:,} images, réparties en trois jeux".replace(",", " "))
    show_formula(
        [
            Term(
                f"{sizes.share(count):.0%}",
                color,
                f"{label} : {count:,} images".replace(",", " "),
            )
            for label, count, color in (
                ("entraînement", sizes.training, explanations.WEIGHT_COLOR),
                ("validation", sizes.validation, explanations.SCORE_COLOR),
                ("test", sizes.test, explanations.OUTPUT_COLOR),
            )
        ]
    )
    explanations.explain(explanations.SPLIT)


def render_curves_slide(model: DigitClassifier, recognition: Recognition) -> None:
    try:
        history = cached_history()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_REPORT.format(error=error))
        return
    last = history[-1]
    text_column, figure_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        key_fact(f"Perte : {history[0].loss:.2f} → {last.loss:.3f} en {last.epoch} époques")
        gap = last.train_accuracy - last.validation_accuracy
        key_fact(
            f"Entraînement {last.train_accuracy:.1%} · validation {last.validation_accuracy:.1%}"
        )
        explanations.explain(f"- Écart de **{gap:.1%}** : léger **surapprentissage**")
        explanations.explain(explanations.TRAINING_CURVES)
    figure = training_history_figure(history).update_layout(height=SLIDE_FIGURE_HEIGHT)
    figure_column.plotly_chart(figure)


def render_test_slide(model: DigitClassifier, recognition: Recognition) -> None:
    try:
        evaluation = cached_evaluation()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_REPORT.format(error=error))
        return
    text_column, figure_column = st.columns(TEXT_SHARE, vertical_alignment="center")
    with text_column:
        source = st.radio("Source des images", list(evaluation), horizontal=True)
        matrix = evaluation[source]
        key_fact(f"{confusion_accuracy(matrix):.1%} de réussite sur des images jamais vues")
        explanations.explain(
            "\n".join(
                f"- **{true} pris pour {answer}** : {times} fois"
                for true, answer, times in top_confusions(matrix, TOP_CONFUSIONS)
            )
        )
        explanations.explain(explanations.CONFUSION)
    figure = confusion_matrix_figure(matrix).update_layout(height=SLIDE_FIGURE_HEIGHT)
    figure_column.plotly_chart(figure)
