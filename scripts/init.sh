#!/usr/bin/env bash
# Prépare un poste après le clone, en une commande, depuis la racine du dépôt :
#   ./scripts/init.sh
# Relançable sans risque : chaque étape saute ce qui est déjà fait.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { printf '\n==> %s\n' "$1"; }
fail() { printf 'Erreur : %s\n' "$1" >&2; exit 1; }

step "Outils"
command -v uv >/dev/null || fail "uv manquant : curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v pnpm >/dev/null || fail "pnpm manquant : npm install -g pnpm (Node 24 requis)"
if ! command -v pre-commit >/dev/null; then
  uv tool install pre-commit
fi

step "Backend : dépendances"
cd "$ROOT_DIR/backend"
uv sync --locked

step "Backend : .env"
if [[ ! -f .env ]]; then
  cp .env.example .env
  secret_key="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(50))')"
  sed -i.bak "s|^DJANGO_SECRET_KEY=.*|DJANGO_SECRET_KEY=${secret_key}|" .env && rm -f .env.bak
  echo "backend/.env créé avec une clé secrète locale."
else
  echo "backend/.env existe déjà, conservé."
fi

step "Backend : migrations"
uv run python manage.py migrate

step "Frontend : dépendances"
cd "$ROOT_DIR/frontend"
pnpm install --frozen-lockfile

step "Garde-fous git (pre-commit)"
cd "$ROOT_DIR"
pre-commit install

printf '\nPrêt. Lancer le backend et le frontend : ./scripts/dev.sh\n'
