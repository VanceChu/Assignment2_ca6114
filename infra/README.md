# Scenario 3 Infra

This directory bootstraps a reusable SDXL LoRA training platform for the `6-a800` host.

## Current Target

- Host alias: `6-a800`
- Remote repo path: `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114`
- Runtime root: `<repo>/runtime`
- Shared training backend: `sd-scripts`
- Shared inference backend: `ComfyUI` (optional for tonight)
- Python runner strategy: project-local `venv`, not new conda envs

## Tonight Quick Start

Run these commands on `6-a800` after the repo is synced:

```bash
bash infra/bin/preflight.sh
bash infra/bin/bootstrap_host.sh
bash infra/bin/create_student_space.sh student_a student_b student_c
cp infra/templates/sdxl_style_lora.env.example runtime/students/student_a/configs/sdxl_style_lora.env
cp infra/templates/sdxl_style_lora.env.example runtime/students/student_b/configs/sdxl_style_lora.env
cp infra/templates/sdxl_style_lora.env.example runtime/students/student_c/configs/sdxl_style_lora.env
```

Then edit each student config and set:

- `STUDENT_ID`
- `TRIGGER_TOKEN`
- `BASE_MODEL`

Prepare data for each student:

- curated images -> `runtime/students/<student_id>/dataset_curated/`
- captions -> `runtime/students/<student_id>/captions/`

Validate and build the training folder:

```bash
bash infra/bin/prepare_dataset.sh student_a
bash infra/bin/prepare_dataset.sh student_b
bash infra/bin/prepare_dataset.sh student_c
```

Smoke test one LoRA:

```bash
bash infra/bin/train_student_lora.sh student_a --gpu 0 --smoke
```

Launch a batch tonight:

```bash
bash infra/bin/batch_train_students.sh --smoke student_a:0 student_b:1 student_c:2
```

## Notes

- `bootstrap_host.sh` clones `sd-scripts` and creates `runtime/venvs/sdscripts`.
- Set `INSTALL_COMFYUI=1` before running `bootstrap_host.sh` if you also want a ready `runtime/venvs/comfyui` environment.
- Each GPU has its own lock file, so one job can run per GPU without collisions.
