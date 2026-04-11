#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python_executable "${SDSCRIPTS_PYTHON}" "sd-scripts python"
ensure_runtime_tree
ensure_workspace_dirs
config_path="$(workspace_config_path)"
if [[ ! -f "${config_path}" ]]; then
  echo "missing config: ${config_path}" >&2
  exit 1
fi

source "${config_path}"
output_csv="${WORKSPACE_ROOT}/report_assets/validation_manifest.csv"
"${SDSCRIPTS_PYTHON}" "${SCRIPT_DIR}/render_validation_manifest.py"   --template "${INFRA_ROOT}/templates/validation_prompts.yaml"   --trigger-token "${TRIGGER_TOKEN}"   --workspace-name "$(workspace_name)"   --output "${output_csv}"

echo "ComfyUI execution is not automated yet; use infra/workflows/sdxl_style_lora_inference.json and this manifest to run validation batches."
