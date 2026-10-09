# Vision

Projet du module DATA732 (visualisation de données).

## Équipe

- Antoine ([@antoine-tena](https://github.com/antoine-tena))
- Nathan ([@NathanBarrachin](https://github.com/NathanBarrachin))
- Pavel ([@Trofymov-Pavlo](https://github.com/Trofymov-Pavlo))

## Organisation

- `backend/` : l'API, en Django + Django Ninja ; règles dans
  [`backend/CLAUDE.md`](backend/CLAUDE.md).
- `frontend/` : l'interface, en Nuxt 4 + Vue 3 ; règles dans
  [`frontend/CLAUDE.md`](frontend/CLAUDE.md).
- [`CLAUDE.md`](CLAUDE.md) : la charte commune (nommage, taille, git…), pour les humains comme
  pour les agents.
- `.vscode/` : réglages et extensions recommandées, partagés par toute l'équipe.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et [pnpm](https://pnpm.io/) (avec Node 24).
Puis, une fois après le clone, depuis la racine du dépôt :

```sh
./scripts/init.sh
```

Le script installe les dépendances du backend et du frontend, crée `backend/.env` avec une
clé secrète locale, applique les migrations et active les garde-fous du commit (pre-commit,
dont le refus de commiter sur `main` ou `develop`). On peut le relancer sans risque.

## Lancer le projet

```sh
./scripts/dev.sh
```

- Frontend : http://localhost:3000 (les appels `/api` sont relayés vers le backend).
- Backend : http://localhost:8000/api/ (documentation de l'API sur `/api/docs`).

## Vérifier le code

```sh
cd backend && uv run ruff check . && uv run mypy . && uv run pytest
cd frontend && pnpm typecheck
```

## Contribuer

1. Créer une branche depuis `main` (`feat/…`, `fix/…`, voir [BRANCHES]).
2. Respecter les règles du [`CLAUDE.md`](CLAUDE.md).
3. Ouvrir une pull request : elle doit être approuvée par un autre membre de l'équipe avant la
   fusion. Personne ne peut pousser directement sur `main` ou `develop`.
