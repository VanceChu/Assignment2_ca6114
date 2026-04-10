#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 student_id [--gpu <id>] [--smoke]" >&2
  exit 1
fi

student_id="$1"
shift
require_student_id "${student_id}"

gpu_id=0
smoke=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --gpu)
      gpu_id="$2"
      shift 2
      ;;
    --smoke)
      smoke=1
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

ensure_runtime_tree
require_python_executable "${SDSCRIPTS_PYTHON}" "sd-scripts python"
config_path="$(student_config_path "${student_id}")"
if [[ ! -f "${config_path}" ]]; then
  cp "${INFRA_ROOT}/templates/sdxl_style_lora.env.example" "${config_path}"
  echo "created config template at ${config_path}; edit it and rerun" >&2
  exit 1
fi

source "${config_path}"
student_dir="$(student_root "${student_id}")"
if [[ "${smoke}" == '1' ]]; then
  bash "${SCRIPT_DIR}/prepare_dataset.sh" "${student_id}" --smoke
else
  bash "${SCRIPT_DIR}/prepare_dataset.sh" "${student_id}"
fi

train_root="${student_dir}/dataset_train"
output_dir="${student_dir}/checkpoints"
log_dir="${student_dir}/logs"
mkdir -p "${output_dir}" "${log_dir}" "${LORAS_ROOT}"

if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "base model missing: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${SDSCRIPTS_ROOT}/sdxl_train_network.py" ]]; then
  echo "sd-scripts is missing; run infra/bin/bootstrap_host.sh first" >&2
  exit 1
fi

max_steps="${MAX_TRAIN_STEPS:-1200}"
save_every="${SAVE_EVERY_N_STEPS:-200}"
if [[ "${smoke}" == '1' ]]; then
  max_steps=100
  save_every=50
fi

warmup_steps="$(python3 -c "import math; print(max(1, math.ceil(${max_steps} * ${WARMUP_RATIO:-0.05})))")"
timestamp="$(date '+%Y%m%d_%H%M%S')"
output_prefix="${OUTPUT_PREFIX:-${student_id}_sdxl_style_lora}"
log_path="${log_dir}/train_${timestamp}_gpu${gpu_id}.log"
lock_path="${LOCK_ROOT}/gpu${gpu_id}.lock"

printf 'gpu=%s
' "${gpu_id}" | tee "${log_path}"
printf 'student=%s
' "${student_id}" | tee -a "${log_path}"
printf 'base_model=%s
' "${BASE_MODEL}" | tee -a "${log_path}"
printf 'train_root=%s
' "${train_root}" | tee -a "${log_path}"
printf 'log=%s
' "${log_path}" | tee -a "${log_path}"

flock "${lock_path}" env CUDA_VISIBLE_DEVICES="${gpu_id}" "${SDSCRIPTS_PYTHON}" "${SDSCRIPTS_ROOT}/sdxl_train_network.py"   --pretrained_model_name_or_path "${BASE_MODEL}"   --train_data_dir "${train_root}"   --resolution "${RESOLUTION:-1024,1024}"   --output_dir "${output_dir}"   --output_name "${output_prefix}"   --save_model_as safetensors   --network_module networks.lora   --network_dim "${NETWORK_DIM:-16}"   --network_alpha "${NETWORK_ALPHA:-16}"   --learning_rate "${LEARNING_RATE:-1e-4}"   --unet_lr "${UNET_LR:-1e-4}"   --text_encoder_lr "${TEXT_ENCODER_LR:-5e-5}"   --lr_scheduler "${LR_SCHEDULER:-cosine}"   --lr_warmup_steps "${warmup_steps}"   --train_batch_size "${BATCH_SIZE:-1}"   --max_train_steps "${max_steps}"   --save_every_n_steps "${save_every}"   --mixed_precision "${MIXED_PRECISION:-bf16}"   --save_precision "${SAVE_PRECISION:-bf16}"   --optimizer_type AdamW8bit   --caption_extension .txt   --cache_latents   --gradient_checkpointing   --xformers 2>&1 | tee -a "${log_path}"
