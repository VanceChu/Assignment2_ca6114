# Scenario 3 Infra

This directory bootstraps a single-workspace SDXL LoRA pipeline for the `6-a800` host.

## Current Target

- Host alias: `6-a800`
- Remote repo path: `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114`
- Runtime root: `<repo>/runtime`
- Working directory: `runtime/workspace/`
- Shared training backend: `sd-scripts`
- Shared inference backend: `ComfyUI` (optional)
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
- Set `INSTALL_COMFYUI=1` before running `bootstrap_host.sh` if you also want `runtime/venvs/comfyui`.
- `train_lora.sh` still uses per-GPU `flock`, so you can safely choose which GPU to occupy.
- The current pipeline is intentionally single-workspace to reduce the learning surface while you first get one LoRA run working end-to-end.
