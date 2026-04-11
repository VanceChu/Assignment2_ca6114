#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_runtime_tree

INSTALL_COMFYUI="${INSTALL_COMFYUI:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

create_venv_if_missing() {
  local venv_dir="$1"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    python3 -m venv "${venv_dir}"
  fi
}

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"
  if [[ ! -d "${target_dir}/.git" ]]; then
    git clone --depth 1 --filter=blob:none "${repo_url}" "${target_dir}"
  fi
}

write_comfyui_model_paths() {
  cat > "${COMFYUI_EXTRA_MODEL_PATHS}" <<EOF
assignment2_ca6114:
  base_path: ${MODELS_ROOT}
  checkpoints: checkpoints
  loras: loras
  vae: vae
EOF
}

install_comfyui_example_workflow() {
  mkdir -p "${COMFYUI_TEMPLATE_WORKFLOWS_ROOT}"
  cat > "${COMFYUI_TEMPLATE_NODE_ROOT}/__init__.py" <<'EOF'
"""Assignment2 ComfyUI workflow templates."""
EOF
  cp "${INFRA_ROOT}/workflows/sdxl_style_lora_inference.json" \
    "${COMFYUI_TEMPLATE_WORKFLOWS_ROOT}/sdxl_style_lora_inference.json"
}

clone_if_missing https://github.com/kohya-ss/sd-scripts "${SDSCRIPTS_ROOT}"
create_venv_if_missing "${SDSCRIPTS_VENV}"

env PIP_INDEX_URL="${PIP_INDEX_URL}" "${SDSCRIPTS_PYTHON}" -m pip install --upgrade pip
"${SDSCRIPTS_PYTHON}" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
(
  cd "${SDSCRIPTS_ROOT}"
  env PIP_INDEX_URL="${PIP_INDEX_URL}" "${SDSCRIPTS_PYTHON}" -m pip install -r requirements.txt
)
env PIP_INDEX_URL="${PIP_INDEX_URL}" "${SDSCRIPTS_PYTHON}" -m pip install bitsandbytes tensorboard pillow pyyaml

if [[ "${INSTALL_COMFYUI}" == "1" ]]; then
  clone_if_missing https://github.com/Comfy-Org/ComfyUI "${COMFYUI_ROOT}"
  create_venv_if_missing "${COMFYUI_VENV}"
  env PIP_INDEX_URL="${PIP_INDEX_URL}" "${COMFYUI_PYTHON}" -m pip install --upgrade pip
  "${COMFYUI_PYTHON}" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
  (
    cd "${COMFYUI_ROOT}"
    env PIP_INDEX_URL="${PIP_INDEX_URL}" "${COMFYUI_PYTHON}" -m pip install -r requirements.txt
  )
  write_comfyui_model_paths
  install_comfyui_example_workflow
fi

printf 'runtime_root=%s
' "${RUNTIME_ROOT}"
printf 'sdscripts_python=%s
' "${SDSCRIPTS_PYTHON}"
printf 'comfyui_python=%s\n' "${COMFYUI_PYTHON}"
printf 'pip_index_url=%s\n' "${PIP_INDEX_URL}"
