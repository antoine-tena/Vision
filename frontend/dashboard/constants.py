"""[ZERO-MAGIE] : textes et réglages du tableau de bord, nommés une seule fois."""

SITE_TITLE = "Vision"
SITE_DESCRIPTION = "IA écrite à la main (NumPy) qui reconnaît les chiffres manuscrits ou imprimés."

# [IDS] et réglages de la zone de dessin.
CANVAS_KEY = "zone-de-dessin"
CANVAS_PIXELS = 280
PEN_WIDTH = 18
# Chiffre blanc sur fond noir, comme dans MNIST.
CANVAS_BACKGROUND = "#000000"
PEN_COLOR = "#ffffff"
CANVAS_BORDER = "#888888"
UPLOAD_TYPES = ("png", "jpg", "jpeg", "bmp", "webp")
PREVIEW_WIDTH = 140

HOME_TITLE = "Reconnaître"
TRAINING_TITLE = "Entraînement"
TRAINING_DESCRIPTION = (
    "Comment le réseau a appris : la perte et l'exactitude époque par époque, sa mesure sur des "
    "images qu'il n'a jamais vues, et les motifs que ses neurones cachés ont appris à guetter."
)
MISSING_REPORT = (
    "Rapport d'entraînement introuvable ou illisible ({error}). Le régénérer depuis backend/ : "
    "`uv run python -m network.train` (ou `network.evaluate` pour la seule évaluation)."
)
PATTERN_GRID_COLUMNS = 16

# Onglets de l'accueil : seul l'onglet ouvert s'exécute ([ETAT]).
INPUT_TABS_KEY = "source-du-chiffre"
DRAWING_TAB = "Dessiner"
EXAMPLES_TAB = "Exemples d'entraînement"
UPLOAD_TAB = "Importer une image"
# Le réseau en direct : la propagation ne se rejoue que pour une nouvelle image.
LAST_ANIMATED_KEY = "dernieres-images-animees"
LIVE_FRAME_SECONDS = 0.08
MISSING_SAMPLE = (
    "Échantillon d'entraînement introuvable ou illisible ({error}). Le régénérer depuis "
    "backend/ : `uv run python -m network.sample`."
)
