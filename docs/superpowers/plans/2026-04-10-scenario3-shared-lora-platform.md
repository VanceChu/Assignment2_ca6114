# Scenario 3 Shared Style LoRA Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one A800-based shared ComfyUI + SDXL style LoRA platform that three students can reuse, while keeping each student's dataset, LoRA weights, outputs, and report evidence isolated.

**Architecture:** Use one remote Ubuntu host as a single-GPU shared platform. Run ComfyUI as the shared inference entrypoint, run `sd-scripts` as the shared LoRA training backend, and serialize GPU-heavy jobs with a lock file so only one training or validation batch owns the A800 at a time. Keep public assets read-only in `shared/` and per-student artifacts isolated in `students/<student_id>/`.

**Tech Stack:** Ubuntu server, SSH, Python 3.10+, Miniconda, ComfyUI, `sd-scripts`, SDXL base model, shell scripts, YAML templates, Markdown docs

---

## Decisions Locked Before Implementation

- Use **Scenario 3 LoRA-only** as the core workflow. Do not add Face Swap or Image Edit in v1.
- Use **one shared platform** with **one independent style LoRA per student**.
- Use **SDXL** as the common base model so all students compare against the same baseline.
- Use **CLI-based training** with `sd-scripts`, not GUI-based training, to keep runs reproducible.
- Keep **ComfyUI private** on the server and access it through SSH tunneling; do not expose it directly on the public network.
- Use **manual per-image captions** plus a shared caption template, not auto-captioning in v1.
- Enforce **single-GPU serialization** with `flock`; no concurrent training jobs on the A800.

## Target Repo Outputs

- `infra/README.md`: deployment and operator guide
- `infra/docs/server_layout.md`: server directory map and permission model
- `infra/docs/student_workflow.md`: step-by-step student usage guide
- `infra/bin/preflight.sh`: verify GPU, disk, Python, and network prerequisites
- `infra/bin/bootstrap_host.sh`: create conda envs, clone dependencies, and prepare directories
- `infra/bin/create_student_space.sh`: create per-student dataset/output/config folders
- `infra/bin/prepare_dataset.sh`: validate image count, caption parity, and image sizes
- `infra/bin/start_comfyui.sh`: start shared ComfyUI service on localhost
- `infra/bin/train_student_lora.sh`: train one student's SDXL style LoRA with fixed defaults
- `infra/bin/run_validation_matrix.sh`: generate the required prompt/seed/weight matrix for one student
- `infra/templates/sdxl_style_lora.env.example`: default LoRA hyperparameters
- `infra/templates/validation_prompts.yaml`: prompt tiers and evaluation set
- `infra/templates/report_evidence_checklist.md`: required screenshots, grids, and notes for the report
- `infra/workflows/sdxl_style_lora_inference.json`: shared ComfyUI inference workflow

## Target Server Layout

Use `/workspace/scenario3-shared-lora` as the only deployment root:

```text
/workspace/scenario3-shared-lora/
  shared/
    bin/
    comfyui/
    models/
      checkpoints/
      loras/
      vae/
    sd-scripts/
    templates/
    workflows/
    logs/
    locks/
  students/
    student_a/
      dataset_raw/
      dataset_curated/
      captions/
      configs/
      checkpoints/
      outputs/
      report_assets/
      logs/
    student_b/
    student_c/
```

## Training Defaults

- Base checkpoint: `sd_xl_base_1.0.safetensors`
- LoRA type: standard SDXL LoRA
- Resolution: `1024`
- Rank / alpha: `16 / 16`
- Batch size: `1`
- Epochs: `10`
- Max train steps: `1200`
- Save checkpoints every `200` steps
- Optimizer: `AdamW8bit`
- UNet LR: `1e-4`
- Text encoder LR: `5e-5`
- LR scheduler: `cosine`
- Warmup: `5%`
- Validation checkpoints to review: `step 400`, `step 800`, `step 1200`
- LoRA trigger token format: `<studentid_style>`

## Validation Matrix Required Per Student

- Prompt tiers: `simple`, `medium`, `complex`
- Seeds: `3 fixed seeds`
- LoRA weights: `0.6`, `0.8`, `1.0`
- Baseline comparison: same prompts and seeds with **no LoRA**
- Required output set: `3 prompt tiers x 3 seeds x 4 settings = 36 images per student`
- Report comparison must cover:
  - base SDXL vs own LoRA
  - simple vs medium vs complex prompts
  - best and worst examples
  - missing features and future improvement ideas

### Task 1: Verify Host and Lock the Deployment Baseline

**Files:**
- Create: `infra/bin/preflight.sh`
- Create: `infra/docs/server_layout.md`
- Modify: `infra/README.md`

- [ ] **Step 1: Write the host preflight script**

The script must collect:
- `hostname`
- `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`
- `df -h /workspace`
- `python3 --version`
- `git --version`
- `curl --version`

- [ ] **Step 2: Run preflight against `interview`**

Run:
```bash
ssh interview 'bash -s' < infra/bin/preflight.sh
```

Expected:
- GPU reports one A800-class device with about 80 GB VRAM
- `/workspace` exists and has enough space for models, datasets, and outputs

- [ ] **Step 3: Lock the deployment root**

Record in `infra/docs/server_layout.md` that all platform files live under:
```text
/workspace/scenario3-shared-lora
```

- [ ] **Step 4: Update operator notes**

Add to `infra/README.md`:
- server alias is `interview`
- ComfyUI will bind to `127.0.0.1:8188`
- users connect through SSH tunnel, not public exposure

### Task 2: Bootstrap Shared Environments and Directories

**Files:**
- Create: `infra/bin/bootstrap_host.sh`
- Modify: `infra/README.md`
- Modify: `infra/docs/server_layout.md`

- [ ] **Step 1: Write the host bootstrap script**

The script must:
- install Miniconda if missing
- create conda env `comfyui-s3`
- create conda env `sdscripts-s3`
- create `/workspace/scenario3-shared-lora/shared/bin`
- create `/workspace/scenario3-shared-lora/shared/templates`
- clone ComfyUI into `/workspace/scenario3-shared-lora/shared/comfyui`
- clone `sd-scripts` into `/workspace/scenario3-shared-lora/shared/sd-scripts`
- copy `infra/bin/*.sh` into `/workspace/scenario3-shared-lora/shared/bin/`
- copy `infra/templates/*` into `/workspace/scenario3-shared-lora/shared/templates/`
- copy `infra/workflows/*` into `/workspace/scenario3-shared-lora/shared/workflows/`
- create `shared/models`, `shared/workflows`, `shared/logs`, `shared/locks`

- [ ] **Step 2: Keep model ownership shared and predictable**

All shared assets must be readable by every student account, but only the platform owner updates them.

- [ ] **Step 3: Run bootstrap once**

Run:
```bash
ssh interview 'bash -s' < infra/bin/bootstrap_host.sh
```

Expected:
- both conda envs created
- repo directories created
- remote `shared/bin` and `shared/templates` contain the platform scripts and templates
- no student-specific data written yet

### Task 3: Create Per-Student Isolated Workspaces

**Files:**
- Create: `infra/bin/create_student_space.sh`
- Modify: `infra/docs/server_layout.md`
- Create: `infra/docs/student_workflow.md`

- [ ] **Step 1: Write the student-space creation script**

For each student id, create:
- `dataset_raw/`
- `dataset_curated/`
- `captions/`
- `configs/`
- `checkpoints/`
- `outputs/`
- `report_assets/`
- `logs/`

- [ ] **Step 2: Standardize student ids**

Use exactly:
- `student_a`
- `student_b`
- `student_c`

- [ ] **Step 3: Run the creation script for all three students**

Run:
```bash
ssh interview '/workspace/scenario3-shared-lora/shared/bin/create_student_space.sh student_a'
ssh interview '/workspace/scenario3-shared-lora/shared/bin/create_student_space.sh student_b'
ssh interview '/workspace/scenario3-shared-lora/shared/bin/create_student_space.sh student_c'
```

Expected:
- all student directories exist
- no files are shared across student directories except read-only model assets

- [ ] **Step 4: Document the student contract**

In `infra/docs/student_workflow.md`, state that each student is responsible for:
- collecting their own images
- writing their own captions
- running their own final validation set
- keeping their own report notes

### Task 4: Define Dataset Rules and Validation

**Files:**
- Create: `infra/bin/prepare_dataset.sh`
- Create: `infra/templates/report_evidence_checklist.md`
- Modify: `infra/docs/student_workflow.md`

- [ ] **Step 1: Fix the dataset standard**

Each student dataset must satisfy:
- `20-30` curated images
- each image corresponds to one caption file
- shortest side at least `1024 px`
- all images belong to one coherent visual style
- no private or copyrighted images unless the student is allowed to use them

- [ ] **Step 2: Write dataset validation logic**

The script must fail if:
- image count is below `20`
- image count is above `30`
- any image lacks a matching `.txt` caption
- any caption lacks the student's trigger token
- any image is smaller than `1024 px` on the shortest side

- [ ] **Step 3: Define caption format**

Use:
```text
<studentid_style>, subject, medium, lighting, color palette, composition
```

Example:
```text
<student_a_style>, city street, watercolor illustration, soft morning light, pastel palette, wide shot
```

- [ ] **Step 4: Add report evidence requirements**

In `infra/templates/report_evidence_checklist.md`, require:
- raw dataset contact sheet
- 3 sample captions
- training config screenshot or config dump
- loss or progress screenshots
- validation grids
- limitation notes

### Task 5: Build the Shared ComfyUI Inference Lane

**Files:**
- Create: `infra/bin/start_comfyui.sh`
- Create: `infra/workflows/sdxl_style_lora_inference.json`
- Modify: `infra/README.md`

- [ ] **Step 1: Keep ComfyUI minimal**

Use only:
- core ComfyUI
- ComfyUI-Manager if needed for dependency handling

Do not add nonessential node packs in v1.

- [ ] **Step 2: Define the shared workflow**

The workflow must include:
- checkpoint loader for SDXL base
- LoRA loader
- positive prompt
- negative prompt
- sampler
- VAE decode
- image save

- [ ] **Step 3: Bind ComfyUI to localhost only**

Run target:
```bash
python main.py --listen 127.0.0.1 --port 8188
```

- [ ] **Step 4: Document access**

Add to `infra/README.md`:
```bash
ssh -L 8188:127.0.0.1:8188 interview
```

Then open `http://127.0.0.1:8188` locally.

### Task 6: Build the Reproducible LoRA Training Wrapper

**Files:**
- Create: `infra/bin/train_student_lora.sh`
- Create: `infra/templates/sdxl_style_lora.env.example`
- Modify: `infra/docs/student_workflow.md`

- [ ] **Step 1: Write the shared default config template**

The template must include:
- student id
- trigger token
- dataset path
- output path
- base model path
- resolution
- rank
- alpha
- batch size
- epochs
- max train steps
- learning rates

- [ ] **Step 2: Serialize GPU access with a lock**

Wrap training with:
```bash
flock /workspace/scenario3-shared-lora/shared/locks/gpu0.lock <training command>
```

- [ ] **Step 3: Standardize output naming**

Output names must follow:
```text
<student_id>_sdxl_style_lora_step<step>.safetensors
```

- [ ] **Step 4: Run one smoke training job**

Use a `5-image` temporary sample set and `100` max steps.

Expected:
- training starts
- checkpoint saves successfully
- GPU lock prevents concurrent training

### Task 7: Build the Validation Matrix and Batch Output Export

**Files:**
- Create: `infra/bin/run_validation_matrix.sh`
- Create: `infra/templates/validation_prompts.yaml`
- Modify: `infra/workflows/sdxl_style_lora_inference.json`

- [ ] **Step 1: Freeze the prompt tiers**

Define exactly three sections in `infra/templates/validation_prompts.yaml`:
- `simple`
- `medium`
- `complex`

- [ ] **Step 2: Freeze the seeds**

Use exactly:
- `101`
- `202`
- `303`

- [ ] **Step 3: Freeze the LoRA weights**

Generate outputs for:
- `baseline` (no LoRA)
- `0.6`
- `0.8`
- `1.0`

- [ ] **Step 4: Export report-ready filenames**

Filename format:
```text
<student_id>__<tier>__seed<seed>__lora<weight>.png
```

- [ ] **Step 5: Run the validation matrix for one student**

Expected:
- `36` images created
- filenames are deterministic
- outputs land in `students/<student_id>/report_assets/validation_matrix/`

### Task 8: Define Report Evidence and Final Acceptance

**Files:**
- Modify: `infra/templates/report_evidence_checklist.md`
- Modify: `infra/README.md`
- Modify: `infra/docs/student_workflow.md`

- [ ] **Step 1: Require each student to keep four comparison sections**

Each report must include:
- dataset and style target
- base SDXL vs LoRA comparison
- simple vs medium vs complex prompt comparison
- current limitations and desired future features

- [ ] **Step 2: Require specific visual evidence**

Each student must retain:
- one dataset overview sheet
- one training-parameter table
- one checkpoint selection note explaining why a step was chosen
- one `36-image` validation matrix
- one short failure-case section

- [ ] **Step 3: Define completion criteria**

The platform is complete only when:
- all three student spaces exist
- one shared ComfyUI workflow runs through SSH tunnel
- one LoRA smoke train succeeds
- one full validation matrix succeeds
- docs are sufficient for another student to follow without verbal explanation

## Test Plan

- Preflight test: host reports expected GPU, disk, and Python availability
- Directory test: all `shared/` and `students/<student_id>/` folders exist
- Dataset test: validator rejects missing captions, small images, and wrong token usage
- Training test: smoke run saves a valid LoRA checkpoint
- Lock test: a second training command blocks or fails cleanly while the first holds the GPU lock
- Inference test: ComfyUI loads base SDXL plus one trained LoRA and saves an image
- Validation test: one student's `36-image` matrix is produced with deterministic names
- Documentation test: a classmate can follow `infra/docs/student_workflow.md` and reach the first successful train without ad hoc operator help

## Assumptions and Defaults

- Assume `interview` is the correct SSH alias for the target Ubuntu host; verify in Task 1.
- Assume the host has one A800-class GPU with about `80 GB` VRAM; verify in Task 1.
- Assume `/workspace` is the preferred large-volume path on the host; if Task 1 disproves this, replace every `/workspace/scenario3-shared-lora` reference consistently before implementation.
- Assume the platform owner performs initial bootstrap and first smoke run.
- Assume each student is responsible for their own dataset legality, style choice, captions, final run, and report writing.
- Default scope is **one base model, one workflow, one LoRA per student**. Do not expand scope before the core path is stable.
