# Vision

Projet d'équipe du module DATA732 (visualisation de données) : Antoine, Nathan, Pavel.

## Démarrage rapide

Après le clone, depuis la racine du dépôt (prérequis : [uv](https://docs.astral.sh/uv/) et
[pnpm](https://pnpm.io/) avec Node 24) :

```sh
uv tool install git+https://github.com/antoine-tena/kiln   # une fois par poste
kiln install
kiln dev
```

Puis ouvrir http://localhost:3000. Détail de chaque commande ci-dessous.

## Équipe

- [@antoine-tena](https://github.com/antoine-tena)
- [@NathanBarrachin](https://github.com/NathanBarrachin)
- [@Trofymov-Pavlo](https://github.com/Trofymov-Pavlo)

## Organisation

- `backend/` : l'API, en Django + Django Ninja ; règles dans [`backend/CLAUDE.md`](backend/CLAUDE.md).
- `frontend/` : l'interface, en Nuxt 4 + Vue 3 ; règles dans [`frontend/CLAUDE.md`](frontend/CLAUDE.md).
- [`CLAUDE.md`](CLAUDE.md) : la charte commune (nommage, taille, git…), pour les humains comme
  pour les agents.
- `.vscode/` : réglages et extensions recommandées, partagés par toute l'équipe.

Le projet est généré et tenu à jour par [kiln](https://github.com/antoine-tena/kiln) (réponses
dans `.copier-answers.yml`). Changer un paramètre (équipe, frameworks, podman…) :
`kiln set <clé>=<valeur>`, sur une branche.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), [pnpm](https://pnpm.io/) (avec Node 24) et l'outil kiln, installé une fois par
poste :

```sh
uv tool install git+https://github.com/antoine-tena/kiln
```

Puis, après le clone, depuis la racine du dépôt :

```sh
kiln install
```

La commande demande une fois le système du poste (Linux, WSL ou macOS), installe les
dépendances (backend, frontend), crée `backend/.env`, applique les migrations et active les garde-fous du
commit (pre-commit, dont le refus de commiter sur `main` ou `develop`). On peut la relancer sans
risque.

## Lancer le projet

```sh
kiln dev
```

- Backend : http://localhost:8000/api/ (documentation de l'API sur `/api/docs`).
- Frontend : http://localhost:3000 (les appels `/api` sont relayés vers le backend).

## Mettre à jour

Sur une branche dédiée (`chore/kiln-update`), avec un arbre propre :

```sh
kiln update            # mises à jour compatibles
kiln update --major    # y compris les versions majeures
```

Elle rapporte les évolutions du modèle (charte, outillage, configuration) sans écraser les
modifications locales, met à jour les dépendances, puis lance les contrôles. Relire le diff,
puis ouvrir une pull request.

## Vérifier le code

```sh
kiln check
```

Ruff, mypy et pytest pour chaque projet Python, ESLint et le contrôle des types côté frontend, puis la charte.

## Contribuer

1. Créer une branche depuis `main` (`feat/…`, `fix/…`, voir [BRANCHES]).
2. Respecter les règles du [`CLAUDE.md`](CLAUDE.md).
3. Ouvrir une pull request : elle doit être approuvée par un autre membre de l'équipe avant la
   fusion. Personne ne peut pousser directement sur `main` ou `develop`.
