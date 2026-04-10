#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 student_id" >&2
  exit 1
fi

student_id="$1"
require_student_id "${student_id}"
require_python_executable "${SDSCRIPTS_PYTHON}" "sd-scripts python"
config_path="$(student_config_path "${student_id}")"
if [[ ! -f "${config_path}" ]]; then
  echo "missing config: ${config_path}" >&2
  exit 1
fi

source "${config_path}"
student_dir="$(student_root "${student_id}")"
output_csv="${student_dir}/report_assets/validation_manifest.csv"
"${SDSCRIPTS_PYTHON}" "${SCRIPT_DIR}/render_validation_manifest.py"   --template "${INFRA_ROOT}/templates/validation_prompts.yaml"   --trigger-token "${TRIGGER_TOKEN}"   --student-id "${student_id}"   --output "${output_csv}"

echo "ComfyUI execution is not automated yet; use infra/workflows/sdxl_style_lora_inference.json and this manifest to run validation batches."
