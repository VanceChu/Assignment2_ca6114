#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

smoke_flag=''
if [[ "${1:-}" == '--smoke' ]]; then
  smoke_flag='--smoke'
  shift
fi

if [[ "$#" -eq 0 ]]; then
  set -- student_a:0 student_b:1 student_c:2
fi

for spec in "$@"; do
  student_id="${spec%%:*}"
  gpu_id="${spec##*:}"
  require_student_id "${student_id}"
  student_dir="$(student_root "${student_id}")"
  mkdir -p "${student_dir}/logs"
  log_path="${student_dir}/logs/batch_launch_$(date '+%Y%m%d_%H%M%S')_gpu${gpu_id}.log"
  if [[ -n "${smoke_flag}" ]]; then
    nohup bash "${SCRIPT_DIR}/train_student_lora.sh" "${student_id}" --gpu "${gpu_id}" --smoke > "${log_path}" 2>&1 &
  else
    nohup bash "${SCRIPT_DIR}/train_student_lora.sh" "${student_id}" --gpu "${gpu_id}" > "${log_path}" 2>&1 &
  fi
  printf 'launched student=%s gpu=%s pid=%s log=%s
' "${student_id}" "${gpu_id}" "$!" "${log_path}"
done
