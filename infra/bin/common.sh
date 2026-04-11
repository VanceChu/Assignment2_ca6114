#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${INFRA_ROOT}/.." && pwd)"
RUNTIME_ROOT="${SCENARIO3_RUNTIME_ROOT:-${PROJECT_ROOT}/runtime}"
SHARED_ROOT="${RUNTIME_ROOT}/shared"
WORKSPACE_ROOT="${RUNTIME_ROOT}/workspace"
MODELS_ROOT="${SHARED_ROOT}/models"
CHECKPOINTS_ROOT="${MODELS_ROOT}/checkpoints"
LORAS_ROOT="${MODELS_ROOT}/loras"
VAE_ROOT="${MODELS_ROOT}/vae"
LOG_ROOT="${SHARED_ROOT}/logs"
LOCK_ROOT="${SHARED_ROOT}/locks"
COMFYUI_ROOT="${SHARED_ROOT}/comfyui"
COMFYUI_CUSTOM_NODES_ROOT="${COMFYUI_ROOT}/custom_nodes"
COMFYUI_TEMPLATE_NODE_NAME="assignment2_ca6114_templates"
COMFYUI_TEMPLATE_NODE_ROOT="${COMFYUI_CUSTOM_NODES_ROOT}/${COMFYUI_TEMPLATE_NODE_NAME}"
COMFYUI_TEMPLATE_WORKFLOWS_ROOT="${COMFYUI_TEMPLATE_NODE_ROOT}/example_workflows"
SDSCRIPTS_ROOT="${SHARED_ROOT}/sd-scripts"
VENV_ROOT="${RUNTIME_ROOT}/venvs"
SDSCRIPTS_VENV="${VENV_ROOT}/sdscripts"
COMFYUI_VENV="${VENV_ROOT}/comfyui"
SDSCRIPTS_PYTHON="${SDSCRIPTS_VENV}/bin/python"
COMFYUI_PYTHON="${COMFYUI_VENV}/bin/python"
COMFYUI_EXTRA_MODEL_PATHS="${COMFYUI_ROOT}/extra_model_paths.yaml"

ensure_runtime_tree() {
  mkdir -p "${CHECKPOINTS_ROOT}" "${LORAS_ROOT}" "${VAE_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}" "${WORKSPACE_ROOT}" "${VENV_ROOT}"
}

ensure_workspace_dirs() {
  mkdir -p     "${WORKSPACE_ROOT}/dataset_raw"     "${WORKSPACE_ROOT}/dataset_curated"     "${WORKSPACE_ROOT}/captions"     "${WORKSPACE_ROOT}/dataset_train"     "${WORKSPACE_ROOT}/configs"     "${WORKSPACE_ROOT}/checkpoints"     "${WORKSPACE_ROOT}/outputs"     "${WORKSPACE_ROOT}/report_assets"     "${WORKSPACE_ROOT}/logs"
}

require_python_executable() {
  local path="$1"
  local label="$2"
  if [[ ! -x "${path}" ]]; then
    echo "${label} is missing: ${path}" >&2
    echo "run infra/bin/bootstrap_host.sh first" >&2
    exit 1
  fi
}

workspace_config_path() {
  printf '%s
' "${WORKSPACE_ROOT}/configs/sdxl_style_lora.env"
}

workspace_name() {
  local configured_name="${WORKSPACE_NAME:-workspace}"
  printf '%s
' "${configured_name}"
}
