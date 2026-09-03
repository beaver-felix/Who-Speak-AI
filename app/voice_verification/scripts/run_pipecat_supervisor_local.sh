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

if [[ "${VOICE_AGENT_CONVERSATION_ENABLED:-false}" != "true" ]]; then
  echo "Set VOICE_AGENT_CONVERSATION_ENABLED=true in ${ENV_FILE} before starting Pipecat."
  exit 1
fi
if [[ -z "${PIPECAT_SUPERVISOR_SECRET:-}" || "${#PIPECAT_SUPERVISOR_SECRET}" -lt 32 ]]; then
  echo "Set PIPECAT_SUPERVISOR_SECRET to a random value of at least 32 characters."
  exit 1
fi

cd "${WORKSPACE_DIR}"
exec python -m app.assistant.pipecat_runtime.session_supervisor
