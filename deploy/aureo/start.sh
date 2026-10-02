#!/usr/bin/env bash
set -euo pipefail
AUREO_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
AUREO_ENV_DIR="${AUREO_ENV_DIR:-/workspace/aureo-env}"
AUREO_GATEWAY_ENV_DIR="${AUREO_GATEWAY_ENV_DIR:-${AUREO_ENV_DIR}-gateway}"
export AUREO_WORKER_PYTHON="${AUREO_WORKER_PYTHON:-${AUREO_ENV_DIR}/bin/python}"
cd "$AUREO_REPO_ROOT"
case "${1:-pod}" in
  pod) exec "$AUREO_GATEWAY_ENV_DIR/bin/python" -m aureo.gpu serve --host 0.0.0.0 --port 8000 ;;
  serverless) exec "$AUREO_GATEWAY_ENV_DIR/bin/python" -m aureo.gpu runpod ;;
  *) echo "Usage: start.sh [pod|serverless]" >&2; exit 2 ;;
esac
