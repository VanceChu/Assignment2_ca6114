#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

printf 'host=%s
' "$(hostname)"
printf 'project_root=%s
' "${PROJECT_ROOT}"
printf 'runtime_root=%s
' "${RUNTIME_ROOT}"
printf 'date=%s
' "$(date '+%Y-%m-%d %H:%M:%S %Z')"

echo '--- gpu ---'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo 'nvidia-smi missing'
fi

echo '--- storage ---'
df -h "${PROJECT_ROOT}"
df -h /mnt/world_foundational_model || true

echo '--- tools ---'
command -v python3 || true
command -v python || true
command -v git || true
command -v conda || true
command -v curl || true
python3 --version || python --version || true
git --version || true
conda --version || true
curl --version | head -n 1 || true
