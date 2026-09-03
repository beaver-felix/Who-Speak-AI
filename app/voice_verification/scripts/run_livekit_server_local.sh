#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${APP_DIR}/.env.local"

if ! command -v livekit-server >/dev/null 2>&1; then
  echo "livekit-server is not installed. Install the local LiveKit server binary, then re-run this script."
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example and configure local-only values."
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -z "${LIVEKIT_API_KEY:-}" || -z "${LIVEKIT_API_SECRET:-}" ]]; then
  echo "Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET in ${ENV_FILE}."
  exit 1
fi
if [[ "${#LIVEKIT_API_SECRET}" -lt 32 ]]; then
  echo "LIVEKIT_API_SECRET must be at least 32 characters; use the same value in every local process."
  exit 1
fi

# Keep the server and gateway/supervisor on exactly the same local credentials.
# `--dev` is suitable only for this loopback development server.
exec livekit-server --dev --keys "${LIVEKIT_API_KEY}: ${LIVEKIT_API_SECRET}"
