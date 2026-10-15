"""Textes et formules de présentation : l'info clé en gros, puis quelques points courts."""

import re

import streamlit as st

from dashboard.formula import Part, Term

WEIGHT_COLOR = "#56B4E9"
PIXEL_COLOR = "#E69F00"
BIAS_COLOR = "#CC79A7"
OUTPUT_COLOR = "#2EC27E"
SCORE_COLOR = "#F0E442"
ACCENT_COLORS = (WEIGHT_COLOR, PIXEL_COLOR, OUTPUT_COLOR, BIAS_COLOR)
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")

NEURON_FORMULA: list[Part] = [
    Term("z", OUTPUT_COLOR, "note du neurone"),
    "=",
    Term("w<sub>1</sub>", WEIGHT_COLOR, "poids : importance du pixel 1"),
    "·",
    Term("x<sub>1</sub>", PIXEL_COLOR, "pixel 1 : 0 = blanc, 1 = noir"),
    "+ … +",
    Term("w<sub>784</sub>x<sub>784</sub>", WEIGHT_COLOR, "pareil pour les 784 pixels"),
    "+",
    Term("b", BIAS_COLOR, "biais : décalage propre au neurone"),
]

RELU_FORMULA: list[Part] = [
    Term("h", OUTPUT_COLOR, "sortie du neurone"),
    "= max(0,",
    Term("z", OUTPUT_COLOR, "sa note"),
    ")",
]

VOTE_FORMULA: list[Part] = [
    Term("vote", SCORE_COLOR, "points donnés au chiffre"),
    "=",
    Term("h", OUTPUT_COLOR, "sortie du neurone caché"),
    "&times;",
    Term("w", WEIGHT_COLOR, "poids : neurone → chiffre"),
]

SOFTMAX_FORMULA: list[Part] = [
    Term("p<sub>k</sub>", OUTPUT_COLOR, "probabilité du chiffre k"),
    "=",
    Term("e<sup>z<sub>k</sub></sup>", SCORE_COLOR, "score du chiffre k, amplifié"),
    "/",
    Term("Σ eᶻ", PIXEL_COLOR, "somme des 10 scores amplifiés"),
]

INPUT_FORMULA: list[Part] = [
    Term("x", PIXEL_COLOR, "l'entrée"),
    "= (",
    Term("x<sub>1</sub>", PIXEL_COLOR, "pixel 1 : 0 = vide, 1 = encre"),
    ", … ,",
    Term("x<sub>784</sub>", PIXEL_COLOR, "pixel 784"),
    ")",
]

EVIDENCE_FORMULA: list[Part] = [
    Term("apport", SCORE_COLOR, "ce que le pixel donne au chiffre"),
    "=",
    Term("x", PIXEL_COLOR, "le pixel"),
    "&times;",
    Term("poids", WEIGHT_COLOR, "son poids total vers le chiffre"),
]

LOSS_FORMULA: list[Part] = [
    Term("L", PIXEL_COLOR, "perte : l'erreur"),
    "= &minus;log(",
    Term("&ycirc;<sub>vrai</sub>", OUTPUT_COLOR, "probabilité donnée au bon chiffre"),
    ")",
]

DESCENT_FORMULA: list[Part] = [
    Term("w", WEIGHT_COLOR, "nouveau poids"),
    "&larr;",
    Term("w", WEIGHT_COLOR, "ancien poids"),
    "&minus;",
    Term("&eta;", BIAS_COLOR, "pas : 0,1 puis décroissant"),
    "&middot;",
    Term("&part;L / &part;w", SCORE_COLOR, "gradient : effet du poids sur l'erreur"),
]

INPUT = """
- **Dessin 280 x 280** ou **image importée** → ramené au **format d'entraînement**, centré
- **Exemples** : déjà en 28 x 28
- **Pas une couche** : aucun calcul, juste 784 nombres
"""

WEIGHT_SIGNS = """
- **Poids positif élevé** : pixel noir → active fort le neurone
- **Poids négatif** : pixel noir → va contre le neurone
- **Poids proche de 0** : le pixel ne compte pas
- Poids tirés au hasard, puis corrigés à chaque erreur : **apprendre = régler les poids**
"""

SHAPES = """
- Image = les 784 poids d'un neurone : **bleu** = encre voulue, **orange** = encre refusée
- **Nom** déduit de la tache bleue (approximation : le réseau ignore ces mots)
"""

LIVE_NETWORK = """
**Récapitulatif, chaque partie et son étape :**
- **Entrée** (pixels) → Étape 1
- **Couche 1** (128 neurones) → Étape 2
- **Traits** (les votes) → Étape 3
- **Couche 2** (les 10 %) → Étape 4, puis Résultat
"""

LIVE_LEGEND = """
- **Gros point clair** = neurone allumé ; petit sombre = éteint
- **Traits** = votes : bleu pour, orange contre
"""

NEURON_VOTES = """
- **« n°35 +2.9 »** : le neurone 35 donne +2.9 points au chiffre
- **Image** = motif guetté, ses 784 poids
- **Bleu** = encre voulue ; **orange** = encre refusée
"""

EVIDENCE = """
- **Bleu** : ce trait pousse vers le chiffre
- **Orange** : ce trait l'en éloigne
- **Blanc** : pixel vide ou sans effet
"""

PROBABILITIES = """
| Chiffre | Score z | eᶻ | Probabilité |
|---|---|---|---|
| 1 | 2 | 7,4 | 12 % |
| 7 | **4** | **54,6** | **87 %** |
| 9 | -1 | 0,4 | 1 % |

**eᶻ** rend tout positif et creuse les écarts, puis **÷ total** (62,4)
"""

SPLIT = """
- 70 000 manuscrites (MNIST) + 7 000 imprimées
- Validation : 10 % mis de côté au départ, jamais appris
- Test : jeu séparé, mesuré une seule fois à la fin
"""

TRAINING_CURVES = r"""
- **Surapprentissage** : le réseau apprend par cœur au lieu de comprendre ; il se voit à l'écart
- Mini-lots de 128 images ; 1 époque = toutes les images ; 10 époques
- Perte : $L = -\log(p_{\text{bon chiffre}})$, doit baisser
- Correction : $W \leftarrow W - \eta \cdot \partial L / \partial W$, pas $\eta = 0{,}1$ décroissant
"""

CONFUSION = """
- Ligne = vrai chiffre ; colonne = réponse ; diagonale = bonnes réponses
- Ces confusions sont intuitives : les formes se ressemblent
"""

HIDDEN_PATTERNS = """
- 1 image = les 784 poids d'un neurone, appris seuls pendant l'entraînement
"""


def key_fact(text: str) -> None:
    """L'information à retenir, en gros ; elle décrit aussi la figure ([FIGURES-ACCESSIBLES])."""
    st.markdown(f"### {text}")


def explain(text: str) -> None:
    """Quelques points courts, sans encadré ; les mots en gras prennent tour à tour une couleur
    de la palette, comme les termes des formules."""
    colors = iter(range(len(ACCENT_COLORS) * len(text)))
    colored = BOLD_PATTERN.sub(
        lambda match: (
            f'<b style="color:{ACCENT_COLORS[next(colors) % len(ACCENT_COLORS)]}">'
            f"{match.group(1)}</b>"
        ),
        text,
    )
    st.markdown(colored, unsafe_allow_html=True)
