# Vision

IA écrite à la main (NumPy) qui reconnaît les chiffres manuscrits ou imprimés.

<!-- This file belongs to the project: kiln wrote it once and never changes it. kiln's rules live
in .claude/rules/kiln/ and are kept current by `kiln update`. Write here what Claude cannot
guess about this project, and keep it under 200 lines: every session reads it. -->

## Repo map

Where to look, so as not to explore: read it before opening files, and update it in the commit
that adds, moves or removes a directory, a domain or an entry point. One line per place, what
it holds; no copy of the code.

- `backend/pipeline/`: loading (`sources.py`), transformations (`transforms.py`), figures
  (`figures.py`); `backend/notebooks/`: marimo notebooks; `backend/data/`: data (`sample/` is
  committed). `backend/tests/`: tests.
- `frontend/app.py`: entry point; `frontend/dashboard/`: pages, layout and constants;
  `frontend/tests/`: tests.
- `docs/backlog/`: the work to do; `charter.toml`: the checked rules; `.claude/rules/kiln/`:
  kiln's rules for Claude.

## Notes

<!-- Decisions, conventions and pitfalls of this project that Claude cannot guess from the code:
"we never call X directly, always through Y", "the staging database is reset every night"… -->

## Carried over from the instructions of kiln 0.3

### From `CLAUDE.md`

#### Carte du repo

  (paquet `figures/`, un module par écran) ; `backend/notebooks/` : carnets marimo ;
  `backend/data/` : données (`sample/`, exemples d'entraînement, commité). `backend/tests/` : tests.
  - `sources.py` : MNIST téléchargé dans `data/raw/` (hors dépôt) ; `printed.py` : chiffres
    imprimés générés depuis les polices du poste ; `dataset.py` : un jeu (784 pixels + chiffre) ;
    `transforms.py` : toute image ramenée au format MNIST ; `recognition.py` : image → prédiction ;
    `reasoning.py` : raisonnement interne (votes des neurones, pixels décisifs) ; `neuron_shapes.py` : nom du motif d'un neurone (trait, boucle…) ; `activity.py` :
    activité de tous les neurones et ordre de propagation, pour le réseau en direct.
- `backend/network/` : réseau de neurones écrit en NumPy, sans framework. `model.py` : le
  classifieur (784 → couche cachée ReLU → 10) ; `layers.py`, `activations.py`, `losses.py`,
  `optimizers.py` : briques ; `training.py` : mini-lots ; `train.py` : entraîne et enregistre
  (`uv run python -m network.train`) ; `evaluate.py` : matrices de confusion du jeu de test ;
  `sample.py` : échantillon commité du jeu d'entraînement.
  Modèle entraîné et ses rapports commités dans `backend/data/models/`.
- `frontend/app.py` : point d'entrée (pages : Présentation, Lexique — `home.py`, `glossary.py`) ;
  `frontend/dashboard/` : diapositives, mise en page et constantes ; `frontend/tests/` : tests.
  - `home.py` : la présentation en diaporama ; `slides.py` : ordre des diapositives et
    navigation (◀ ▶, clavier) ; `sections.py` : diapositives de section avec leur plan.
  - Parties : `project.py` (le projet), `learning.py` et `training.py` (l'apprentissage),
    `choice.py` (choisir un chiffre : dessin `canvas.py`, `examples.py`, import `images.py`),
    `reasoning.py` (étapes 1 à 4, résultat), `live.py` (conclusion, le réseau en direct).
  - `formula.py` : formules fléchées ; `explanations.py` : textes et formules ; `loaders.py` :
    modèle et rapports en cache ; `constants.py` : textes et réglages.

### From `backend/CLAUDE.md`

(identifiants, régimes, hiérarchie), les règles communes (§1), les standards 2026 (§4), Git
(§5) et les règles des données et des figures (§6) sont dans le `CLAUDE.md` de la racine.

### From `frontend/CLAUDE.md`

(identifiants, régimes, hiérarchie), les règles communes (§1), les standards 2026 (§4), Git
(§5) et les règles des données et des figures (§6) sont dans le `CLAUDE.md` de la racine.
