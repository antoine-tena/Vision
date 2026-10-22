"""Textes et formules de présentation : l'info clé en gros, puis quelques points courts."""

import re

import streamlit as st

from dashboard.formula import Part, Term

# Une couleur par notion, la même dans les formules, les textes et le schéma du réseau.
PIXEL_COLOR = "#E69F00"
WEIGHT_COLOR = "#56B4E9"
HIDDEN_COLOR = "#2EC27E"
OUTPUT_COLOR = "#CC79A7"
SCORE_COLOR = "#E9C46A"
BIAS_COLOR = "#B39DDB"
LOSS_COLOR = "#FF7B72"
RATE_COLOR = "#9AA5B1"
NEUTRAL_COLOR = "#E6EDF3"
DEFAULT_BOLD_COLOR = SCORE_COLOR
LIST_COLORS = (
    WEIGHT_COLOR,
    HIDDEN_COLOR,
    PIXEL_COLOR,
    OUTPUT_COLOR,
    SCORE_COLOR,
    BIAS_COLOR,
    LOSS_COLOR,
)
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
# Mot en gras → couleur de sa notion (premier mot-clé trouvé) ; sinon gras sans couleur.
KEYWORD_COLORS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("bleu",), WEIGHT_COLOR),
    (("orange",), PIXEL_COLOR),
    (("blanc",), NEUTRAL_COLOR),
    (("biais",), BIAS_COLOR),
    (("perte", "erreur", "gradient", "surapprentissage", "entropie", "pente"), LOSS_COLOR),
    (("poids", "entraînement", "apprentissage"), WEIGHT_COLOR),
    (("pixel", "entrée", "image", "dessin", "exemples", "784", "format"), PIXEL_COLOR),
    (
        ("neurone", "couche 1", "motif", "trait", "boucle", "tache", "diagonale", "n°", "128"),
        HIDDEN_COLOR,
    ),
    (("vote", "score", "eᶻ", "total"), SCORE_COLOR),
    (("sortie", "couche 2", "chiffre", "probabilit", "softmax", "test", "llm"), OUTPUT_COLOR),
    (("validation",), SCORE_COLOR),
)

NEURON_FORMULA: list[Part] = [
    Term("z", HIDDEN_COLOR, "note du neurone"),
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
    Term("h", HIDDEN_COLOR, "sortie du neurone"),
    "= max(0,",
    Term("z", HIDDEN_COLOR, "sa note"),
    ")",
]

VOTE_FORMULA: list[Part] = [
    Term("vote", SCORE_COLOR, "points donnés au chiffre"),
    "=",
    Term("h", HIDDEN_COLOR, "sortie du neurone caché"),
    "&times;",
    Term("w", WEIGHT_COLOR, "poids : neurone → chiffre"),
]

SOFTMAX_FORMULA: list[Part] = [
    Term("p<sub>k</sub>", OUTPUT_COLOR, "probabilité du chiffre k"),
    "=",
    Term("e<sup>z<sub>k</sub></sup>", SCORE_COLOR, "score du chiffre k, amplifié"),
    "/",
    Term("Σ eᶻ", SCORE_COLOR, "somme des 10 scores amplifiés"),
]

FAMILY_FORMULA: list[Part] = [
    Term("IA", SCORE_COLOR, "le grand domaine"),
    "&rarr;",
    Term("apprentissage", WEIGHT_COLOR, "il apprend sur des exemples"),
    "&rarr;",
    Term("réseau de neurones", HIDDEN_COLOR, "même famille que les LLM"),
    "&rarr;",
    Term("classifieur", OUTPUT_COLOR, "une image → un chiffre parmi 10"),
]

ARCHITECTURE_FORMULA: list[Part] = [
    Term("784", PIXEL_COLOR, "pixels d'entrée"),
    "&rarr;",
    Term("128", HIDDEN_COLOR, "neurones cachés"),
    "&rarr;",
    Term("10", OUTPUT_COLOR, "neurones de sortie"),
]

SIZE_FORMULA: list[Part] = [
    Term("~100 000", WEIGHT_COLOR, "paramètres de Vision"),
    "&lt;&lt;",
    Term("des milliards", SCORE_COLOR, "paramètres d'un LLM"),
]

LOOP_FORMULA: list[Part] = [
    Term("propagation", OUTPUT_COLOR, "il voit une image et prédit"),
    "&rarr;",
    Term("perte", LOSS_COLOR, "on mesure son erreur"),
    "&rarr;",
    Term("rétropropagation", LOSS_COLOR, "le sens de correction de chaque poids"),
    "&rarr;",
    Term("descente", WEIGHT_COLOR, "chaque poids bouge d'un petit pas"),
]

TWO_LEVELS_FORMULA: list[Part] = [
    Term("784 pixels", PIXEL_COLOR, "l'image"),
    "&rarr;",
    Term("784 poids", WEIGHT_COLOR, "dessinent le motif"),
    "&rarr;",
    Term("neurone", HIDDEN_COLOR, "détecte son motif"),
    "&rarr;",
    Term("10 poids", WEIGHT_COLOR, "pour quels chiffres il compte"),
    "&rarr;",
    Term("10 chiffres", OUTPUT_COLOR, "les sorties"),
]

TWO_LEVELS = """
- **Poids d'entrée** : leur ensemble est le motif
- **Poids de sortie** : **bleu** pousse vers le chiffre, **orange** l'en éloigne
"""

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
    Term("L", LOSS_COLOR, "perte : l'erreur"),
    "= &minus;log(",
    Term("&ycirc;<sub>vrai</sub>", OUTPUT_COLOR, "probabilité donnée au bon chiffre"),
    ")",
]

DESCENT_FORMULA: list[Part] = [
    Term("w", WEIGHT_COLOR, "nouveau poids"),
    "&larr;",
    Term("w", WEIGHT_COLOR, "ancien poids"),
    "&minus;",
    Term("&eta;", RATE_COLOR, "pas : 0,1 puis décroissant"),
    "&middot;",
    Term("&part;L / &part;w", LOSS_COLOR, "gradient : effet du poids sur l'erreur"),
]

INPUT = """
- **Dessin 280 x 280** ou **image importée** → ramené au **format d'entraînement**, centré
- **Exemples** : déjà en 28 x 28
- **Pas une couche** : aucun calcul, juste 784 nombres
"""

WEIGHT_SIGNS = """
- **Poids positif** : « cette entrée me pousse à m'activer »
- **Poids négatif** : « cette entrée m'en empêche »
- **Poids proche de 0** : « cette entrée ne m'intéresse pas »
- **Plus il est grand** (en positif ou en négatif), plus l'entrée compte
"""

SHAPES = """
- Image = les 784 poids d'un neurone : **bleu** = encre voulue, **orange** = encre refusée
- **Nom** déduit de la tache bleue (approximation : le réseau ignore ces mots)
"""

EMERGENCE = """
- **Fixé** : 128 neurones reliés aux 784 pixels, poids de départ **au hasard**
- **L'entraînement** pousse chacun vers ce qui est utile → il se **spécialise**
- **Départs différents** → motifs différents ; **autre graine** → autres motifs, même précision
- **Les 10 sorties** : rôle imposé par les étiquettes (le n°7 devient « le 7 »)
"""

LIVE_NETWORK = """
**Récapitulatif, chaque partie et son étape :**
- **Entrée** (pixels) → 3.2
- **Couche 1** (128 neurones) → 3.3
- **Traits** (les votes) → 3.4
- **Couche 2** (les 10 %) → 3.5, puis 3.6
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
- **Entraînement** : les poids apprennent dessus (MNIST manuscrits + chiffres imprimés)
- **Validation** : 10 % mis de côté au départ, mesurés à chaque époque, jamais appris
- **Test** : jeu séparé, mesuré une seule fois à la fin : le vrai score
"""

TRAINING_CURVES = """
- **Surapprentissage** : apprendre par cœur au lieu de comprendre ; il se voit à l'écart
"""

CONFUSION = """
- Ligne = vrai chiffre ; colonne = réponse ; diagonale = bonnes réponses
- Ces confusions sont intuitives : les formes se ressemblent
"""


def key_fact(text: str) -> None:
    """L'information à retenir, en gros et centrée ; elle décrit aussi la figure
    ([ACCESSIBLE-FIGURES])."""
    st.markdown(f"<h3 style='text-align:center'>{text}</h3>", unsafe_allow_html=True)


def explain(text: str) -> None:
    """Quelques points courts, sans encadré ; chaque mot en gras prend la couleur de sa notion."""
    st.markdown(BOLD_PATTERN.sub(colored_bold, text), unsafe_allow_html=True)


def explain_list(text: str) -> None:
    """Une liste (un plan) : chaque mot en gras prend une couleur différente, dans l'ordre."""
    colors = iter(LIST_COLORS * len(text))
    st.markdown(
        BOLD_PATTERN.sub(
            lambda match: f'<b style="color:{next(colors)}">{match.group(1)}</b>', text
        ),
        unsafe_allow_html=True,
    )


def colored_bold(match: re.Match[str]) -> str:
    words = match.group(1)
    lowered = words.lower()
    for keywords, color in KEYWORD_COLORS:
        if any(keyword in lowered for keyword in keywords):
            return f'<b style="color:{color}">{words}</b>'
    return f'<b style="color:{DEFAULT_BOLD_COLOR}">{words}</b>'
