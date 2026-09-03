#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${APP_DIR}/../.." && pwd)"
ENV_FILE="${APP_DIR}/.env.local"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example and configure local-only values."
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

cd "${WORKSPACE_DIR}"
if command -v lk >/dev/null 2>&1; then
  # `lk agent dev` watches the workspace. Local Whisper/HuggingFace model cache
  # writes can therefore trigger a reload and kill the active room job. Use the
  # stable worker mode by default; opt into hot reload only while editing code.
  if [[ "${VOICE_AGENT_HOT_RELOAD:-false}" == "true" ]]; then
    exec lk agent dev app/assistant/livekit_agent.py
  fi
  exec lk agent start app/assistant/livekit_agent.py
fi

echo "LiveKit CLI (lk) is not installed; using the compatible Python CLI without hot reload."
python -m app.assistant.livekit_agent start
