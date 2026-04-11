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
execution_csv="${WORKSPACE_ROOT}/report_assets/validation_execution.csv"
asset_env="${WORKSPACE_ROOT}/report_assets/comfyui_assets.env"
checkpoint_root="${CHECKPOINT_OUTPUT_ROOT:-/mnt/kai_ckp/model/Assignment2_ca6114}"
workspace_checkpoint_dir="${checkpoint_root}/$(workspace_name)"
output_prefix="${OUTPUT_PREFIX:-$(workspace_name)_sdxl_style_lora}"
comfyui_port="${COMFYUI_PORT:-8188}"
comfyui_url="${COMFYUI_URL:-http://127.0.0.1:${comfyui_port}}"
comfyui_log="${WORKSPACE_ROOT}/logs/comfyui_$(date '+%Y%m%d_%H%M%S').log"
started_comfyui=0
comfyui_pid=''

wait_for_comfyui() {
  local attempt
  for attempt in $(seq 1 30); do
    if curl --silent --fail "${comfyui_url}/system_stats" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

start_comfyui_if_needed() {
  if curl --silent --fail "${comfyui_url}/system_stats" >/dev/null 2>&1; then
    return 0
  fi
  require_python_executable "${COMFYUI_PYTHON}" "ComfyUI python"
  bash "${SCRIPT_DIR}/start_comfyui.sh" > "${comfyui_log}" 2>&1 &
  comfyui_pid="$!"
  started_comfyui=1
  if ! wait_for_comfyui; then
    echo "ComfyUI failed to become ready; see ${comfyui_log}" >&2
    if [[ -n "${comfyui_pid}" ]]; then
      kill "${comfyui_pid}" >/dev/null 2>&1 || true
    fi
    exit 1
  fi
}

"${SDSCRIPTS_PYTHON}" "${SCRIPT_DIR}/render_validation_manifest.py" \
  --template "${INFRA_ROOT}/templates/validation_prompts.yaml" \
  --trigger-token "${TRIGGER_TOKEN}" \
  --workspace-name "$(workspace_name)" \
  --output "${output_csv}"

"${SDSCRIPTS_PYTHON}" "${SCRIPT_DIR}/publish_comfyui_assets.py" \
  --base-model-path "${BASE_MODEL}" \
  --lora-dir "${workspace_checkpoint_dir}" \
  --output-prefix "${output_prefix}" \
  --checkpoints-root "${CHECKPOINTS_ROOT}" \
  --loras-root "${LORAS_ROOT}" \
  --workspace-name "$(workspace_name)" \
  --output-env "${asset_env}"

source "${asset_env}"

if [[ "${COMFYUI_AUTO_START:-1}" == '1' ]]; then
  start_comfyui_if_needed
fi

"${SDSCRIPTS_PYTHON}" "${SCRIPT_DIR}/run_comfyui_batch.py" \
  --manifest "${output_csv}" \
  --workflow-template "${INFRA_ROOT}/workflows/sdxl_style_lora_inference.json" \
  --server-url "${comfyui_url}" \
  --base-model-name "${BASE_MODEL_NAME}" \
  --lora-name "${LORA_NAME}" \
  --output-report "${execution_csv}"

printf 'validation_manifest=%s\n' "${output_csv}"
printf 'validation_execution=%s\n' "${execution_csv}"
printf 'comfyui_url=%s\n' "${comfyui_url}"
printf 'comfyui_started=%s\n' "${started_comfyui}"
