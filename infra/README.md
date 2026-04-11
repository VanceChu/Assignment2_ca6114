# Scenario 3 Infra

This directory bootstraps a single-workspace SDXL LoRA pipeline for the `6-a800` host.

## Current Target

- Host alias: `6-a800`
- Remote repo path: `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114`
- Runtime root: `<repo>/runtime`
- Working directory: `runtime/workspace/`
- Checkpoint output root: `/mnt/kai_ckp/model/Assignment2_ca6114/<workspace>/`
- Shared training backend: `sd-scripts`
- Shared inference backend: `ComfyUI`
- Python runner strategy: project-local `venv`, not new conda envs

## One-Person Quick Start

Run these commands on `6-a800`:

```bash
bash infra/bin/preflight.sh
bash infra/bin/bootstrap_host.sh
bash infra/bin/create_workspace.sh
cp infra/templates/sdxl_style_lora.env.example runtime/workspace/configs/sdxl_style_lora.env
```

Then edit `runtime/workspace/configs/sdxl_style_lora.env` and set:

- `TRIGGER_TOKEN`
- `BASE_MODEL`

Put your data in:

- `runtime/workspace/dataset_curated/`
- `runtime/workspace/captions/`

Validate the dataset:

```bash
bash infra/bin/prepare_dataset.sh
```

Run a smoke train:

```bash
bash infra/bin/train_lora.sh --gpu 0 --smoke
```

Run a full train:

```bash
bash infra/bin/train_lora.sh --gpu 0
```

Generate the validation manifest:

```bash
bash infra/bin/run_validation_matrix.sh
```

This now:

- regenerates `runtime/workspace/report_assets/validation_manifest.csv`
- publishes the base model and latest LoRA into `runtime/shared/models/`
- auto-starts ComfyUI on `127.0.0.1:8188` when needed
- queues the full validation batch through the ComfyUI HTTP API
- writes execution results to `runtime/workspace/report_assets/validation_execution.csv`

## Current Workspace Layout

```text
runtime/
  shared/
    sd-scripts/
    comfyui/
    logs/
    locks/
    models/
  workspace/
    dataset_curated/
    captions/
    dataset_train/
    configs/
    checkpoints/
    outputs/
    report_assets/
    logs/
  venvs/
    sdscripts/
    comfyui/
```

## Notes

- `bootstrap_host.sh` clones `sd-scripts` and creates `runtime/venvs/sdscripts`.
- `bootstrap_host.sh` now installs ComfyUI by default and writes `runtime/shared/comfyui/extra_model_paths.yaml` so ComfyUI loads shared checkpoints and LoRAs.
- `bootstrap_host.sh` also registers `sdxl_style_lora_inference.json` as a ComfyUI custom-node example workflow, so it appears in the UI `Templates` browser.
- `train_lora.sh` still uses per-GPU `flock`, so you can safely choose which GPU to occupy.
- LoRA checkpoints are written to `/mnt/kai_ckp/model/Assignment2_ca6114/<workspace>/`, not `runtime/workspace/checkpoints/`.
- `train_lora.sh` publishes the latest trained LoRA as `runtime/shared/models/loras/<workspace>_latest.safetensors`.
- The current pipeline is intentionally single-workspace to reduce the learning surface while you first get one LoRA run working end-to-end.
