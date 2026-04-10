#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_runtime_tree

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 student_a [student_b ...]" >&2
  exit 1
fi

for student_id in "$@"; do
  require_student_id "${student_id}"
  student_dir="$(student_root "${student_id}")"
  mkdir -p     "${student_dir}/dataset_raw"     "${student_dir}/dataset_curated"     "${student_dir}/captions"     "${student_dir}/dataset_train"     "${student_dir}/configs"     "${student_dir}/checkpoints"     "${student_dir}/outputs"     "${student_dir}/report_assets"     "${student_dir}/logs"
  printf 'created=%s
' "${student_dir}"
done
