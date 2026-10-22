"""Réglages partagés par les figures ([ACCESSIBLE-FIGURES])."""

from network.model import DIGIT_COUNT

# Palette Okabe-Ito, lisible par les daltoniens.
COLORBLIND_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
DIGITS = [str(digit) for digit in range(DIGIT_COUNT)]
# Divergente : bleu = pousse vers le chiffre, orange = l'en éloigne, blanc = neutre.
DIVERGING_COLORSCALE = [
    [0.0, COLORBLIND_PALETTE[5]],
    [0.5, "#FFFFFF"],
    [1.0, COLORBLIND_PALETTE[0]],
]
