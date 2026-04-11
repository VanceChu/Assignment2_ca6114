#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_runtime_tree
ensure_workspace_dirs

printf 'created=%s
' "${WORKSPACE_ROOT}"
