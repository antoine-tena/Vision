"""Explications simples et formules affichées sous chaque figure, pour présenter le projet."""

import streamlit as st

INPUT = r"""
**L'entrée.** L'image est d'abord réduite à 28 x 28 = 784 pixels, comme les images qui ont
servi à l'entraînement. Chaque pixel devient un nombre : 0 pour le blanc, 1 pour l'encre noire,
entre les deux pour le gris. Le réseau ne voit que cette liste de 784 nombres :
$x = (x_1, x_2, \dots, x_{784})$. Ce n'est pas une couche : il n'y a aucun calcul ici.
"""

HIDDEN_LAYER = r"""
**La couche 1.** Elle a 128 neurones. Chacun regarde les 784 pixels et leur donne une note
avec ses poids, puis la ReLU éteint les notes négatives :

$$h = \max(0,\ x \cdot W_1 + b_1)$$

Un neurone dont la note est positive est **allumé** : il a trouvé son motif dans l'image. Les
autres restent à 0 et ne comptent plus pour la suite. C'est le groupe de neurones allumés qui
distingue un chiffre d'un autre.
"""

LIVE_NETWORK = r"""
**Comment lire ce schéma.** Le réseau a **deux couches** qui calculent ; l'entrée n'en est pas
une, ce sont juste les pixels de l'image.

- **Entrée** : l'image est réduite à 28 x 28 = 784 pixels ; chaque pixel devient un nombre,
  0 pour le blanc, 1 pour l'encre : $x = (x_1, \dots, x_{784})$.
- **Couche 1, cachée** : 128 neurones. Chacun donne une note à l'image entière, puis la ReLU
  éteint les notes négatives :
  $h = \max(0,\ x \cdot W_1 + b_1)$.
  Un gros point clair est un neurone allumé (il a reconnu son motif), un petit point sombre un
  neurone éteint (note ramenée à 0).
- **Couche 2, sortie** : 10 neurones, un par chiffre. Chacun additionne les votes des neurones
  cachés, puis la softmax change ces 10 scores en pourcentages :
  $p = \mathrm{softmax}(h \cdot W_2 + b_2)$.
- **Les traits** : les votes des neurones allumés pour le chiffre gagnant, en bleu s'ils
  poussent vers lui, en orange s'ils poussent contre.
"""

PROBABILITIES = r"""
**La softmax.** La couche de sortie donne 10 scores, un par chiffre, qui peuvent être n'importe
quels nombres. La softmax les change en probabilités entre 0 et 1, dont la somme fait 100 % :

$$p_k = \frac{e^{z_k}}{e^{z_0} + e^{z_1} + \dots + e^{z_9}}$$

L'exponentielle $e^z$ creuse les écarts : un score un peu plus grand que les autres prend
presque tout. Le chiffre reconnu est celui qui a la plus grande probabilité.
"""

WEIGHT = r"""
**Qu'est-ce qu'un poids ?** Un **poids** est un nombre qui dit **combien un pixel compte** pour
un neurone, comme un barème : un poids positif veut dire « de l'encre ici rapporte des
points », un poids négatif « de l'encre ici en fait perdre », un poids nul « cet endroit ne
compte pas ». La note d'un neurone est la somme, pixel par pixel :

$$\text{note} = \text{pixel}_1 \times \text{poids}_1 + \dots + \text{pixel}_{784} \times
\text{poids}_{784} + \text{biais}$$

Personne ne choisit ces nombres : ils sont tirés au hasard au départ, puis corrigés un peu à
chaque erreur pendant l'entraînement. Le réseau en a environ 100 000 ; **apprendre, c'est
uniquement régler ses poids**.

**Cette carte** montre, pour chaque pixel de l'image, ce qu'il a apporté au score du chiffre :
$\text{apport} = \text{pixel} \times \text{poids total du pixel vers ce chiffre}$ (à travers
les neurones allumés). Bleu : ce trait ressemble au chiffre ; orange : il l'en éloigne ; blanc :
pixel vide ou sans effet.
"""

NEURON_VOTES = r"""
**Les votes des neurones.** « n°35 +2.9 » veut dire : le neurone caché numéro 35 (sur 128) a
donné **+2.9 points** au chiffre. Son vote vaut :

$$\text{vote} = \text{sortie du neurone} \times \text{poids qui le relie au chiffre}$$

Positif, il pousse vers le chiffre ; négatif, contre. Le score du chiffre est la somme de tous
les votes, plus un biais. Chaque petite image est le **motif** que guette le neurone, c'est-à-
dire ses 784 poids rangés en 28 x 28 : en bleu où il veut de l'encre, en orange où il n'en veut
pas. Le neurone s'allume quand le dessin ressemble à son motif.
"""

TRAINING_CURVES = r"""
**Comment le réseau apprend.** On lui montre des milliers de chiffres dont on connaît la
réponse, par paquets de 128 (un **mini-lot**). Un passage sur toutes les images est une
**époque** ; il y en a 10.

- **La perte** mesure à quel point il se trompe : $L = -\log(p_{\text{bon chiffre}})$. Elle
  vaut 0 s'il donne 100 % au bon chiffre et grandit quand il se trompe. Elle doit baisser.
- **La correction** (descente de gradient) : chaque poids est poussé un petit peu dans le sens
  qui fait baisser la perte, $W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$, où
  $\eta$ (le pas) vaut 0,1 au départ et diminue peu à peu.
- **L'exactitude** est le pourcentage d'images bien reconnues. Celle de **validation** est
  mesurée sur des images mises de côté, jamais apprises : c'est elle qui dit si le réseau
  généralise ou s'il apprend par cœur.
"""

CONFUSION = r"""
**La matrice de confusion.** Le test se fait sur des images que le réseau n'a **jamais vues**.
Chaque ligne est le vrai chiffre, chaque colonne la réponse du réseau : la diagonale compte les
bonnes réponses, une case hors diagonale montre une confusion (un 4 pris pour un 9, par
exemple). Exactitude = bonnes réponses ÷ nombre d'images.
"""

HIDDEN_PATTERNS = r"""
**Les motifs appris.** Chaque petite image montre les 784 poids d'un neurone caché, rangés en
28 x 28. Personne ne les a dessinés : ils sont sortis tout seuls de l'entraînement. Certains
ressemblent à des morceaux de chiffres (un trait, une boucle), d'autres à des mélanges.
"""

SHAPES = r"""
**Comment on nomme un motif.** Le réseau ne connaît pas les mots « boucle » ou « diagonale » :
c'est une lecture faite après coup sur les poids de chaque neurone. On garde ses poids les plus
forts (là où il veut de l'encre) et on regarde leur forme : allongée dans un sens, c'est un
**trait** (horizontal, vertical) ou une **diagonale** ; en anneau autour d'un creux, une
**boucle** ; ramassée, une **tache**. La zone (en haut, en bas, à gauche…) est l'endroit où ces
poids se concentrent. C'est une approximation : les motifs appris sont souvent flous.
"""


def explain(text: str) -> None:
    """Encadré d'explication sous une figure."""
    st.info(text)
