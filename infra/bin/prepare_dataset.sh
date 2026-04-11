#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python_executable "${SDSCRIPTS_PYTHON}" "sd-scripts python"

min_images=20
max_images=30
if [[ "${1:-}" == '--smoke' ]]; then
  min_images=5
  max_images=30
fi

ensure_runtime_tree
ensure_workspace_dirs
config_path="$(workspace_config_path)"
if [[ ! -f "${config_path}" ]]; then
  echo "missing config: ${config_path}" >&2
  echo "copy infra/templates/sdxl_style_lora.env.example into that path first" >&2
  exit 1
fi

source "${config_path}"
"${SDSCRIPTS_PYTHON}" "${SCRIPT_DIR}/prepare_dataset.py"   --student-root "${WORKSPACE_ROOT}"   --trigger-token "${TRIGGER_TOKEN}"   --repeats "${REPEATS:-10}"   --min-images "${min_images}"   --max-images "${max_images}"
