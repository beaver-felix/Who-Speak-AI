#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

if [[ -z "${VOICE_MATCHER_TOKEN:-}" ]]; then
  echo "Set VOICE_MATCHER_TOKEN in .env.local before starting the matcher." >&2
  exit 1
fi

exec python -m uvicorn apps.matcher_api.main:app --host 127.0.0.1 --port 8011
