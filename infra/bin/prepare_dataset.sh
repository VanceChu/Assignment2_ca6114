#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 student_id [--smoke]" >&2
  exit 1
fi

student_id="$1"
shift
require_student_id "${student_id}"
require_python_executable "${SDSCRIPTS_PYTHON}" "sd-scripts python"

min_images=20
max_images=30
if [[ "${1:-}" == '--smoke' ]]; then
  min_images=5
  max_images=30
fi

config_path="$(student_config_path "${student_id}")"
if [[ ! -f "${config_path}" ]]; then
  echo "missing config: ${config_path}" >&2
  echo "copy infra/templates/sdxl_style_lora.env.example into that path first" >&2
  exit 1
fi

source "${config_path}"
student_dir="$(student_root "${student_id}")"
"${SDSCRIPTS_PYTHON}" "${SCRIPT_DIR}/prepare_dataset.py"   --student-root "${student_dir}"   --trigger-token "${TRIGGER_TOKEN}"   --repeats "${REPEATS:-10}"   --min-images "${min_images}"   --max-images "${max_images}"
