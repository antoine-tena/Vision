"""Page « Lexique » : les mots, les lettres des formules et les questions d'oral."""

import streamlit as st

from dashboard.constants import GLOSSARY_TITLE
from dashboard.explanations import key_fact

KEY_WORDS = """
1. :blue[**Poids** (w)] : l'importance de chaque connexion ; un nombre par liaison
2. :violet[**Biais** (b)] : le seuil de déclenchement d'un neurone
3. :orange[**Activation, ReLU**] : éteint le neurone si sa note est négative
4. :red[**Perte** (L)] : la mesure de l'erreur
5. :green[**Rétropropagation et descente de gradient**] : l'erreur sert à corriger les poids
"""

LETTERS = """
| Lettre | Sens | Origine |
|---|---|---|
| **x** | une entrée : un pixel, de 0 à 1 | |
| :blue[**w**, **W**] | un poids ; W = tous les poids d'une couche | *weight* |
| :violet[**b**] | un biais | *bias* |
| **z** | la note du neurone, avant activation | |
| :orange[**h**] (ou a) | la sortie du neurone : max(0, z) | *hidden* / *activation* |
| **ŷ** | la prédiction : les probabilités | |
| **y** | la bonne réponse | |
| :red[**L**] | la perte | *loss* |
| :green[**∂L/∂w**] | le gradient : effet d'un poids sur l'erreur | « d rond L sur d rond w » |
| :green[**η**] | le pas d'apprentissage (0,1) | « êta » |
"""

WHOLE_NETWORK = r"""
$$h = \max(0,\ \color{#1f77b4}{W_1} \cdot x + \color{#9467bd}{b_1})
\qquad \hat{y} = \mathrm{softmax}(\color{#1f77b4}{W_2} \cdot h + \color{#9467bd}{b_2})$$
"""

COUNTS = """
| Couche | Neurones | Poids qui entrent | Biais |
|---|---|---|---|
| Entrée (pixels) | 784 | | |
| Cachée : :orange[les détecteurs de motifs] | 128 | :blue[784 x 128 = 100 352] | :violet[128] |
| Sortie : :green[un neurone par chiffre] | 10 | :blue[128 x 10 = 1 280] | :violet[10] |
| **Total** | 138 | :blue[**101 632**] | :violet[**138**] |

**101 770 paramètres** appris. Les 784 poids d'entrée d'un neurone :orange[dessinent son motif] ;
ses 10 poids de sortie disent :green[pour quels chiffres ce motif compte]. Personne ne choisit
les motifs : partis du hasard, ils émergent de l'entraînement.
"""

SOFTMAX = """
:orange[**Softmax**] : on rend chaque note positive (eᶻ), puis on divise par le total →
des probabilités qui font 100 %. :red[**Perte**] = -log(probabilité du bon chiffre) : 0 si
100 %, énorme si le réseau est sûr de lui et se trompe.
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
    st.subheader("Combien de neurones, de poids, de biais")
    st.markdown(COUNTS)
    st.subheader("De la note à la probabilité, puis à l'erreur")
    st.markdown(SOFTMAX)
    st.subheader("Les lettres des formules")
    st.markdown(LETTERS)
    st.subheader("Questions probables à l'oral")
    st.markdown(QUESTIONS)
