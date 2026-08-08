#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=backend exec .venv/bin/uvicorn app.main:app --host "${YWA_HOST:-127.0.0.1}" --port "${YWA_PORT:-8000}"
