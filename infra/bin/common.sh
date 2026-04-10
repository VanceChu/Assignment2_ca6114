#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${INFRA_ROOT}/.." && pwd)"
RUNTIME_ROOT="${SCENARIO3_RUNTIME_ROOT:-${PROJECT_ROOT}/runtime}"
SHARED_ROOT="${RUNTIME_ROOT}/shared"
STUDENTS_ROOT="${RUNTIME_ROOT}/students"
MODELS_ROOT="${SHARED_ROOT}/models"
CHECKPOINTS_ROOT="${MODELS_ROOT}/checkpoints"
LORAS_ROOT="${MODELS_ROOT}/loras"
VAE_ROOT="${MODELS_ROOT}/vae"
LOG_ROOT="${SHARED_ROOT}/logs"
LOCK_ROOT="${SHARED_ROOT}/locks"
COMFYUI_ROOT="${SHARED_ROOT}/comfyui"
SDSCRIPTS_ROOT="${SHARED_ROOT}/sd-scripts"
VENV_ROOT="${RUNTIME_ROOT}/venvs"
SDSCRIPTS_VENV="${VENV_ROOT}/sdscripts"
COMFYUI_VENV="${VENV_ROOT}/comfyui"
SDSCRIPTS_PYTHON="${SDSCRIPTS_VENV}/bin/python"
COMFYUI_PYTHON="${COMFYUI_VENV}/bin/python"

ensure_runtime_tree() {
  mkdir -p "${CHECKPOINTS_ROOT}" "${LORAS_ROOT}" "${VAE_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}" "${STUDENTS_ROOT}" "${VENV_ROOT}"
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

require_student_id() {
  local student_id="${1:-}"
  case "${student_id}" in
    student_a|student_b|student_c) ;;
    *)
      echo "student_id must be one of: student_a, student_b, student_c" >&2
      exit 1
      ;;
  esac
}

student_root() {
  printf '%s
' "${STUDENTS_ROOT}/${1}"
}

student_config_path() {
  printf '%s
' "$(student_root "$1")/configs/sdxl_style_lora.env"
}
