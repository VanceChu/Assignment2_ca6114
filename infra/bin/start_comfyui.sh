#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python_executable "${COMFYUI_PYTHON}" "ComfyUI python"
if [[ ! -f "${COMFYUI_ROOT}/main.py" ]]; then
  echo "ComfyUI is missing; run infra/bin/bootstrap_host.sh first" >&2
  exit 1
fi

port="${COMFYUI_PORT:-8188}"
extra_model_paths_args=()
if [[ -f "${COMFYUI_EXTRA_MODEL_PATHS}" ]]; then
  extra_model_paths_args+=(--extra-model-paths-config "${COMFYUI_EXTRA_MODEL_PATHS}")
fi

"${COMFYUI_PYTHON}" "${COMFYUI_ROOT}/main.py" --listen 127.0.0.1 --port "${port}" "${extra_model_paths_args[@]}"
