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
TOP_SHAPES = 5
# Diaporama de l'accueil : une diapositive par écran, sans défiler.
SLIDE_KEY = "diapositive"
RECOGNITION_KEY = "chiffre-reconnu"
SLIDE_FIGURE_HEIGHT = 390
SLIDE_IMAGE_WIDTH = 260
# Diaporama : sous la barre de Streamlit (60 px), titres centrés, images centrées, texte lisible.
SLIDE_PAGE_STYLE = (
    "<style>"
    "[data-testid=stMainBlockContainer]{padding-top:4.5rem;padding-bottom:0.5rem;"
    "max-width:1500px;margin:0 auto}"
    "[data-testid=stHeading]{text-align:center}"
    "[data-testid=stFullScreenFrame]:has([data-testid=stImage])"
    "{display:flex;justify-content:center}"
    "[data-testid=stImageCaption]{text-align:center}"
    "[data-testid=stMarkdownContainer] li{font-size:1.15rem}"
    ".st-key-raccourcis-clavier{display:none}"
    "</style>"
)
PROJECT_TITLE = "Le projet"
LEARNING_TITLE = "Comment il apprend"
GLOSSARY_TITLE = "Lexique"
TOP_CONFUSIONS = 3
PREVIEW_WIDTH = 160
SECTION_FIGURE_HEIGHT = 370
SHORTCUTS_KEY = "raccourcis-clavier"
EXTRA_NEXT_SHORTCUTS = ("Enter", "Space")
