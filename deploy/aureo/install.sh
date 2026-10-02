#!/usr/bin/env bash
set -euo pipefail
# Dependencies only. Provisioning model weights is a separate explicit step.
AUREO_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
AUREO_ENV_DIR="${AUREO_ENV_DIR:-/workspace/aureo-env}"
AUREO_GATEWAY_ENV_DIR="${AUREO_GATEWAY_ENV_DIR:-${AUREO_ENV_DIR}-gateway}"
AUREO_PYTHON="${AUREO_PYTHON:-python3.11}"
AUREO_INSTALL_MODE="${1:-gpu}"
if [[ "$AUREO_INSTALL_MODE" != "gpu" && "$AUREO_INSTALL_MODE" != "--build" && "$AUREO_INSTALL_MODE" != "--dry-run" ]]; then
  echo "Usage: install.sh [--build|--dry-run]" >&2
  exit 2
fi
if [[ "$AUREO_INSTALL_MODE" == "--dry-run" ]]; then
  echo "Plan: two isolated Python 3.11 venvs (GPU and gateway); uv 0.12.19; hash-locked dependencies; no model downloads."
  exit 0
fi
if [[ "$AUREO_INSTALL_MODE" == "gpu" ]]; then
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null; then
    echo "No NVIDIA GPU available; no environment or model downloads performed." >&2
    exit 3
  fi
fi
if ! command -v "$AUREO_PYTHON" >/dev/null; then
  echo "Install Python 3.11 and venv support first." >&2
  exit 2
fi
"$AUREO_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"'
if [[ "$(realpath -m "$AUREO_ENV_DIR")" == "$(realpath -m "$AUREO_GATEWAY_ENV_DIR")" ]]; then
  echo "GPU and gateway environments must be distinct." >&2
  exit 2
fi
for AUREO_TARGET_ENV in "$AUREO_ENV_DIR" "$AUREO_GATEWAY_ENV_DIR"; do
  if [[ "$(realpath -m "$AUREO_TARGET_ENV")" == "$AUREO_REPO_ROOT/.venv" || "$(realpath -m "$AUREO_TARGET_ENV")" == "$AUREO_REPO_ROOT" ]]; then
    echo "Choose isolated environments outside the application environment." >&2
    exit 2
  fi
  if [[ ! -e "$AUREO_TARGET_ENV" ]]; then
    "$AUREO_PYTHON" -m venv "$AUREO_TARGET_ENV"
  elif [[ ! -f "$AUREO_TARGET_ENV/pyvenv.cfg" ]]; then
    echo "Existing target is not a virtual environment; refusing to overwrite." >&2
    exit 2
  fi
  if [[ ! -x "$AUREO_TARGET_ENV/bin/uv" ]]; then
    "$AUREO_TARGET_ENV/bin/python" -m ensurepip --upgrade
    "$AUREO_TARGET_ENV/bin/python" -m pip install 'uv==0.12.19'
  fi
  AUREO_TARGET_LOCK="$AUREO_REPO_ROOT/deploy/aureo/requirements.lock"
  if [[ "$AUREO_TARGET_ENV" == "$AUREO_GATEWAY_ENV_DIR" ]]; then
    AUREO_TARGET_LOCK="$AUREO_REPO_ROOT/deploy/aureo/requirements-gateway.lock"
  fi
  "$AUREO_TARGET_ENV/bin/uv" --no-config pip sync --python "$AUREO_TARGET_ENV/bin/python" --require-hashes "$AUREO_TARGET_LOCK"
  "$AUREO_TARGET_ENV/bin/uv" --no-config pip check --python "$AUREO_TARGET_ENV/bin/python"
done
if [[ "$AUREO_INSTALL_MODE" == "gpu" ]]; then
  cd "$AUREO_REPO_ROOT"
  "$AUREO_ENV_DIR/bin/python" -m aureo.gpu probe
fi
echo "AUREO dependency environment installed. Model provisioning remains a separate explicit command."
