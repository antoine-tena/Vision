"""Page « Lexique » : les mots, les lettres des formules et les questions d'oral."""

import streamlit as st

from dashboard.constants import GLOSSARY_TITLE
from dashboard.explanations import key_fact

KEY_WORDS = """
1. **Poids** (w) : l'importance de chaque connexion
2. **Biais** (b) : le seuil de déclenchement d'un neurone
3. **Activation, ReLU** : éteint le neurone si sa note est négative
4. **Perte** (L) : la mesure de l'erreur
5. **Rétropropagation et descente de gradient** : l'erreur sert à corriger les poids
"""

LETTERS = """
| Lettre | Sens | Origine |
|---|---|---|
| **x** | une entrée : un pixel, de 0 à 1 | |
| **w**, **W** | un poids ; W = tous les poids d'une couche | *weight* |
| **b** | un biais | *bias* |
| **z** | la note du neurone, avant activation | |
| **h** (ou a) | la sortie du neurone : max(0, z) | *hidden* / *activation* |
| **ŷ** | la prédiction : les probabilités | |
| **y** | la bonne réponse | |
| **L** | la perte | *loss* |
| **∂L/∂w** | le gradient : effet d'un poids sur l'erreur | « d rond L sur d rond w » |
| **η** | le pas d'apprentissage (0,1) | « êta » |
"""

WHOLE_NETWORK = r"""
$$h = \max(0,\ W_1 \cdot x + b_1) \qquad \hat{y} = \mathrm{softmax}(W_2 \cdot h + b_2)$$
"""

QUESTIONS = """
**Pourquoi ReLU ?** Calcul très simple, et le gradient ne s'évanouit pas.

**Pourquoi softmax et entropie croisée ensemble ?** Leur gradient combiné est simple :
probabilités - bonne réponse.

**Pourquoi un dessin peut-il être mal reconnu ?** Trait, épaisseur et centrage diffèrent de
MNIST : d'où le prétraitement qui ramène toute image au format d'entraînement.

**Comment l'améliorer ?** Un réseau convolutif (CNN, ~99 %), des images déformées en plus
(augmentation), du *dropout*, un optimiseur comme Adam.

**Pourquoi 128 neurones cachés ?** Seul nombre choisi librement : plus = plus précis mais
plus lent.
"""


def render_glossary() -> None:
    st.title(GLOSSARY_TITLE)
    key_fact("Les 5 mots à savoir expliquer")
    st.markdown(KEY_WORDS)
    st.subheader("Tout le réseau en deux lignes")
    st.markdown(WHOLE_NETWORK)
    st.subheader("Les lettres des formules")
    st.markdown(LETTERS)
    st.subheader("Questions probables à l'oral")
    st.markdown(QUESTIONS)
