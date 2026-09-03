#!/usr/bin/env bash
set -euo pipefail

WEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "${WEB_DIR}/node_modules" ]]; then
  echo "Web dependencies are missing. Run: cd ${WEB_DIR} && npm install"
  exit 1
fi

cd "${WEB_DIR}"
exec npm run dev
