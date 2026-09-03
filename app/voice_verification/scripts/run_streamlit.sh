#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

if [[ -z "${VOICE_MATCHER_TOKEN:-}" || -z "${VOICE_MODEL_CHECKPOINT:-}" ]]; then
  echo "Set VOICE_MATCHER_TOKEN and VOICE_MODEL_CHECKPOINT in .env.local first." >&2
  exit 1
fi

exec python -m streamlit run streamlit_app.py
