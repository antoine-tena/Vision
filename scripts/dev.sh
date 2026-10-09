#!/usr/bin/env bash
# Lance le backend (http://localhost:8000) et le frontend (http://localhost:3000) ensemble.
# Ctrl+C arrête les deux.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

trap 'kill 0' EXIT
(cd "$ROOT_DIR/backend" && uv run python manage.py runserver) &
(cd "$ROOT_DIR/frontend" && pnpm dev) &
wait
