"""Page « Le projet » : ce qu'est Vision, son architecture, ses résultats, le lien avec les LLM."""

import streamlit as st
from pipeline.sources import DatasetFormatError
from pipeline.transforms import confusion_accuracy

from dashboard.constants import MISSING_REPORT, PROJECT_TITLE
from dashboard.explanations import key_fact
from dashboard.loaders import cached_evaluation, cached_model

PITCH = (
    "Un réseau de neurones écrit à la main en NumPy, sans PyTorch ni TensorFlow, "
    "qui reconnaît les chiffres manuscrits ou imprimés."
)

WHAT_IT_IS = """
- **IA** → **apprentissage automatique** : il apprend sur des exemples, sans règles écrites
- → **réseau de neurones** : même famille que les LLM, en version minuscule
- → **classifieur d'images** : une image → une catégorie parmi 10 (les chiffres 0 à 9)
- **Pas un LLM** : un LLM traite du texte, avec des milliards de paramètres
- **Rien de caché** : propagation, rétropropagation et descente de gradient codées dans le projet
"""

LAYERS = (
    ("Entrée", "784 pixels", "28 x 28, aucun calcul"),
    ("Couche 1, cachée", "128 neurones", "détecteurs de motifs (seul nombre choisi librement)"),
    ("Couche 2, sortie", "10 neurones", "un par chiffre, puis softmax → probabilités"),
)

LLM_COMPARISON = """
| | **Vision** | **Un LLM** |
|---|---|---|
| Entrée | 784 pixels | du texte découpé en tokens |
| Sortie | 1 chiffre parmi 10 | le mot suivant, parmi ~100 000 |
| Paramètres | ~100 000 | des milliards |
| Architecture | réseau dense | Transformer (« attention ») |
| Principes | poids, biais, perte, rétropropagation | **les mêmes** |
"""


def render_project() -> None:
    st.title(PROJECT_TITLE)
    key_fact(PITCH)
    render_figures()
    st.subheader("Ce que c'est")
    st.markdown(WHAT_IT_IS)
    st.subheader("L'architecture : 784 → 128 → 10")
    for column, (name, size, role) in zip(st.columns(len(LAYERS)), LAYERS, strict=True):
        column.metric(name, size)
        column.caption(role)
    st.subheader("Le lien avec les LLM")
    st.markdown(LLM_COMPARISON)


def render_figures() -> None:
    """Les chiffres à connaître par cœur, lus dans le modèle et son évaluation."""
    try:
        evaluation = cached_evaluation()
    except (OSError, DatasetFormatError) as error:
        st.error(MISSING_REPORT.format(error=error))
        return
    columns = st.columns(len(evaluation) + 1)
    parameters = f"{cached_model().parameter_count:,}".replace(",", " ")
    columns[0].metric("Paramètres (poids + biais)", parameters)
    for column, (source, matrix) in zip(columns[1:], evaluation.items(), strict=True):
        column.metric(f"Test sur les {source}", f"{confusion_accuracy(matrix):.1%}")
