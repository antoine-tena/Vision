"""Explications courtes et formules affichées sous chaque figure, pour présenter le projet."""

import streamlit as st

INPUT = r"""
- Dessin : 280 x 280 = 78 400 pixels ; image importée : toute taille
- Ramenée au format d'entraînement : chiffre centré, **28 x 28 = 784 pixels**
- Exemples d'entraînement : déjà en 28 x 28
- Pixel → nombre : 0 = blanc, 1 = encre
- $x = (x_1, \dots, x_{784})$
- Pas une couche : aucun calcul
"""

HIDDEN_LAYER = r"""
- 128 neurones, chacun reçoit les 784 pixels
- Chaque pixel arrive avec son propre **poids**
- Puis ReLU : $z$ gardé s'il est positif, sinon 0 → $h = \max(0, z)$
- $h > 0$ : neurone **allumé** (motif trouvé) ; sinon éteint
"""

# Somme pondérée d'un neurone caché, chaque terme annoté d'une flèche (KaTeX).
NEURON_FORMULA = r"""
z \;=\;
\underset{\substack{\uparrow \\ \text{poids du pixel 1} \\ \text{pour ce neurone}}}{w_1}
\cdot
\underset{\substack{\uparrow \\ \text{pixel 1} \\ \text{0 = blanc, 1 = noir}}}{x_1}
\;+\;
\underset{\substack{\uparrow \\ \text{poids du pixel 2}}}{w_2}
\cdot
\underset{\substack{\uparrow \\ \text{pixel 2}}}{x_2}
\;+\; \dots \;+\;
w_{784} \cdot x_{784}
\;+\;
\underset{\substack{\uparrow \\ \text{biais :} \\ \text{décalage du neurone}}}{b}
"""

WEIGHT_SIGNS = r"""
**Le poids w : son signe et sa taille donnent son effet**
- **Positif élevé** : pixel noir → active fort le neurone
- **Négatif** : pixel noir → va contre le neurone
- **Proche de 0** : ce pixel ne compte pas pour ce neurone
"""

SHAPES = r"""
- Noms déduits des poids les plus forts de chaque neurone
- Allongé → **trait** / **diagonale** ; anneau → **boucle** ; ramassé → **tache**
- Approximation : le réseau ne connaît pas ces mots
"""

LIVE_NETWORK = r"""
- **Entrée** : 784 pixels, $x$
- **Couche 1** : 128 neurones, $h = \max(0,\ x \cdot W_1 + b_1)$
- **Couche 2** : 10 chiffres, $p = \mathrm{softmax}(h \cdot W_2 + b_2)$
- Gros point clair = allumé ; petit sombre = éteint
- Traits : votes, bleu = pour, orange = contre
"""

PROBABILITIES = r"""
- 10 scores → 10 probabilités, total 100 %
- $p_k = \dfrac{e^{z_k}}{e^{z_0} + \dots + e^{z_9}}$
- L'exponentielle creuse les écarts : le meilleur score prend presque tout
"""

WEIGHT = r"""
- **Poids** = combien un pixel compte pour un neurone
- Positif : encre voulue ; négatif : encre refusée ; 0 : sans effet
- Note = $\sum \text{pixel} \times \text{poids} + \text{biais}$
- Hasard au départ, corrigés à chaque erreur : **apprendre = régler les poids** (~100 000)
- Carte : apport de chaque pixel ; bleu = pour, orange = contre
"""

NEURON_VOTES = r"""
- « n°35 +2.9 » : neurone 35 donne +2.9 points au chiffre
- $\text{vote} = \text{sortie du neurone} \times \text{poids vers le chiffre}$
- Score du chiffre = somme des votes + biais
- Image = motif guetté (784 poids en 28 x 28) ; bleu = encre voulue, orange = refusée
"""

TRAINING_CURVES = r"""
- Mini-lots de 128 images ; 1 époque = toutes les images ; 10 époques
- Perte : $L = -\log(p_{\text{bon chiffre}})$, doit baisser
- Correction : $W \leftarrow W - \eta \cdot \partial L / \partial W$
- Pas $\eta = 0{,}1$, puis décroissant
- Validation : images jamais apprises → vrai niveau du réseau
"""

CONFUSION = r"""
- Images jamais vues
- Ligne = vrai chiffre ; colonne = réponse
- Diagonale = bonnes réponses ; hors diagonale = confusions
"""

HIDDEN_PATTERNS = r"""
- 1 image = les 784 poids d'un neurone
- Appris seuls pendant l'entraînement
"""


def explain(text: str) -> None:
    """Encadré d'explication sous une figure."""
    st.info(text)


def describe(text: str) -> None:
    """Texte alternatif d'une figure ([FIGURES-ACCESSIBLES]), replié pour ne pas encombrer."""
    with st.expander("Description du graphique"):
        st.caption(text)
