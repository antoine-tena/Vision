"""Textes et formules de présentation : l'info clé en gros, puis quelques points courts."""

import streamlit as st

from dashboard.formula import Part, Term

WEIGHT_COLOR = "#56B4E9"
PIXEL_COLOR = "#E69F00"
BIAS_COLOR = "#CC79A7"
OUTPUT_COLOR = "#2EC27E"
SCORE_COLOR = "#F0E442"

NEURON_FORMULA: list[Part] = [
    Term("z", OUTPUT_COLOR, "note du neurone"),
    "=",
    Term("w<sub>1</sub>", WEIGHT_COLOR, "poids : importance du pixel 1"),
    "·",
    Term("x<sub>1</sub>", PIXEL_COLOR, "pixel 1 : 0 = blanc, 1 = noir"),
    "+ … +",
    Term("w<sub>784</sub>·x<sub>784</sub>", WEIGHT_COLOR, "pareil pour les 784 pixels"),
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
    Term("Σ e<sup>z</sup>", PIXEL_COLOR, "somme des 10 scores amplifiés"),
]

INPUT = """
- Dessin 280 x 280 ou image importée → ramené au format d'entraînement, chiffre centré
- Exemples d'entraînement : déjà en 28 x 28
- Pas une couche : aucun calcul, juste 784 nombres
"""

WEIGHT_SIGNS = """
- **Poids positif élevé** : pixel noir → active fort le neurone
- **Poids négatif** : pixel noir → va contre le neurone
- **Poids proche de 0** : le pixel ne compte pas
- Poids tirés au hasard, puis corrigés à chaque erreur : **apprendre = régler les poids**
"""

SHAPES = """
- Noms déduits des poids les plus forts de chaque neurone : allongé → trait, anneau → boucle
- Approximation : le réseau ne connaît pas ces mots
"""

LIVE_NETWORK = """
**Où chaque partie est détaillée plus bas :**
- **Entrée** (pixels, à gauche) → **Étape 1**
- **Couche 1** (128 neurones) → **Étape 2**
- **Traits** entre la couche 1 et la sortie (les votes) → **Étape 3**
- **Couche 2** (les 10 pourcentages, à droite) → **Étape 4**, puis **Résultat**

- Gros point clair = neurone allumé ; petit sombre = éteint
- Traits : votes, bleu = pour, orange = contre
"""

NEURON_VOTES = """
- « n°35 +2.9 » : le neurone 35 donne +2.9 points au chiffre
- Image = motif guetté (ses 784 poids) ; bleu = encre voulue, orange = refusée
"""

EVIDENCE = """
- Apport de chaque pixel au score ; bleu = pour, orange = contre
"""

PROBABILITIES = """
- L'exponentielle creuse les écarts : le meilleur score prend presque tout
- Total des 10 probabilités = 100 %
"""

SPLIT = """
- 70 000 manuscrites (MNIST) + 7 000 imprimées
- Validation : 10 % mis de côté au départ, jamais appris
- Test : jeu séparé, mesuré une seule fois à la fin
"""

TRAINING_CURVES = r"""
- Mini-lots de 128 images ; 1 époque = toutes les images ; 10 époques
- Perte : $L = -\log(p_{\text{bon chiffre}})$, doit baisser
- Correction : $W \leftarrow W - \eta \cdot \partial L / \partial W$, pas $\eta = 0{,}1$ décroissant
"""

CONFUSION = """
- Ligne = vrai chiffre ; colonne = réponse ; diagonale = bonnes réponses
"""

HIDDEN_PATTERNS = """
- 1 image = les 784 poids d'un neurone, appris seuls pendant l'entraînement
"""


def key_fact(text: str) -> None:
    """L'information à retenir, en gros ; elle décrit aussi la figure ([FIGURES-ACCESSIBLES])."""
    st.markdown(f"### {text}")


def explain(text: str) -> None:
    """Quelques points courts, sans encadré."""
    st.markdown(text)
