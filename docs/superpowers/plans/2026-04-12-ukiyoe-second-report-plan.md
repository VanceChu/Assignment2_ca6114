# Ukiyo-e Second Scenario 3 Report Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the existing Scenario 3 LoRA + ComfyUI infrastructure to produce a second, independently defensible Ukiyo-e style experiment and report without overwriting the completed Van Gogh assets.

**Architecture:** Keep the codebase and shared infrastructure unchanged, but isolate the second experiment with a new runtime root, a new trigger token, a new workspace name, and a new checkpoint output directory. Use the same training and validation pipeline so the comparison structure stays consistent, while the dataset, LoRA, validation outputs, and report evidence remain clearly separate.

**Tech Stack:** `sd-scripts`, `ComfyUI`, SDXL base checkpoint, shared tokenizer cache, automated validation matrix, CSV evidence, PDF report

---

## File / Directory Map

**Repo files already in use:**
- `infra/bin/bootstrap_host.sh`
- `infra/bin/create_workspace.sh`
- `infra/bin/prepare_dataset.sh`
- `infra/bin/train_lora.sh`
- `infra/bin/run_validation_matrix.sh`
- `infra/templates/sdxl_style_lora.env.example`
- `infra/templates/report_evidence_checklist.md`
- `infra/workflows/sdxl_style_lora_inference.json`

**New experiment runtime root:**
- `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe`

**Key runtime paths for Report B:**
- `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/dataset_curated/`
- `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/captions/`
- `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/configs/sdxl_style_lora.env`
- `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/report_assets/validation_manifest.csv`
- `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/report_assets/validation_execution.csv`
- `/mnt/kai_ckp/model/Assignment2_ca6114/ukiyoe/`

**Van Gogh experiment to preserve untouched:**
- `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime/workspace/...`
- `/mnt/kai_ckp/model/Assignment2_ca6114/vangogh/`

---

### Task 1: Freeze Scope And Separation

**Files:**
- Read: `week5_assignment_summary.md`
- Read: `infra/README.md`
- Read: `infra/docs/student_workflow.md`

- [ ] **Step 1: Lock the report strategy**

Use:
- Report A: Van Gogh
- Report B: Ukiyo-e

Success condition:
- Different style family
- Different dataset
- Different trigger token
- Different checkpoint directory
- Different validation output set

- [ ] **Step 2: Lock the Ukiyo-e substyle**

Do not mix every possible Ukiyo-e look.

Recommended:
- Edo-period woodblock landscape / travel-scene prints

Good artists to draw from:
- Katsushika Hokusai
- Utagawa Hiroshige

Success condition:
- Same medium feel
- Similar color treatment
- Similar contour/flat-plane composition

- [ ] **Step 3: Lock the experiment identifiers**

Use:
- `WORKSPACE_NAME=ukiyoe`
- `TRIGGER_TOKEN='<ukiyoe_style>'`
- `OUTPUT_PREFIX=ukiyoe_sdxl_style_lora`

Success condition:
- No overlap with `vangogh`

---

### Task 2: Create A New Isolated Runtime

**Files:**
- Create: `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/...`
- Create: `/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/configs/sdxl_style_lora.env`

- [ ] **Step 1: Set the second runtime root**

Run on `6-a800`:

```bash
export SCENARIO3_RUNTIME_ROOT=/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe
```

Expected:
- All workspace actions go to `runtime_ukiyoe`, not `runtime`

- [ ] **Step 2: Create the second workspace**

Run:

```bash
bash infra/bin/create_workspace.sh
```

Expected:
- `runtime_ukiyoe/workspace/` created with dataset/config/report directories

- [ ] **Step 3: Create the config from template**

Run:

```bash
cp infra/templates/sdxl_style_lora.env.example /mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe/workspace/configs/sdxl_style_lora.env
```

Expected:
- Config file exists for the Ukiyo-e run

- [ ] **Step 4: Edit only the Ukiyo-e identifiers**

Set:

```dotenv
WORKSPACE_NAME=ukiyoe
TRIGGER_TOKEN='<ukiyoe_style>'
OUTPUT_PREFIX=ukiyoe_sdxl_style_lora
BASE_MODEL=/mnt/kai_ckp/model/Assignment2_ca6114/sd_xl_base_1.0.safetensors
```

Expected:
- Checkpoints will land under `/mnt/kai_ckp/model/Assignment2_ca6114/ukiyoe/`

---

### Task 3: Prepare A Public-Domain Ukiyo-e Dataset

**Files:**
- Populate: `runtime_ukiyoe/workspace/dataset_curated/`
- Populate: `runtime_ukiyoe/workspace/captions/`

- [ ] **Step 1: Choose 20-30 images**

Use:
- `20-30` curated images
- shortest side `>=1024`
- public-domain works
- one coherent substyle only

Recommended sources:
- WikiArt
- The Met
- Rijksmuseum
- National Gallery of Art

Avoid:
- mixed anime-style derivatives
- mixed poster scans with text margins
- mixed portrait + landscape + bird-and-flower if style consistency breaks

- [ ] **Step 2: Keep the dataset narrow**

Recommended visual characteristics:
- black contour lines
- flat color planes
- limited palette
- woodblock print texture
- landscape or city/travel scenes

Success condition:
- A random 6-image contact sheet looks like one family

- [ ] **Step 3: Write one caption per image**

Caption format:

```text
<ukiyoe_style>, ukiyo-e woodblock print, [subject], flat color planes, strong contour lines, Japanese Edo-period composition
```

Example:

```text
<ukiyoe_style>, ukiyo-e woodblock print, riverside bridge with travelers, flat color planes, strong contour lines, Japanese Edo-period composition
```

- [ ] **Step 4: Spot-check three captions**

Verify:
- token included
- style phrase included
- subject differs per image
- wording stays concise and style-first

---

### Task 4: Validate The Dataset

**Files:**
- Build: `runtime_ukiyoe/workspace/dataset_train/`

- [ ] **Step 1: Export the second runtime root again in the shell**

Run:

```bash
export SCENARIO3_RUNTIME_ROOT=/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_ukiyoe
```

- [ ] **Step 2: Run dataset validation**

Run:

```bash
bash infra/bin/prepare_dataset.sh
```

Expected:
- `training_data=.../dataset_train/10_<ukiyoe_style>`
- `image_count=20-30`

- [ ] **Step 3: Stop and fix if validation fails**

Typical failure reasons:
- image too small
- missing `.txt`
- token missing
- too many/few images

Success condition:
- clean validator pass before any training

---

### Task 5: Run Smoke And Full Training

**Files:**
- Output: `/mnt/kai_ckp/model/Assignment2_ca6114/ukiyoe/`
- Logs: `runtime_ukiyoe/workspace/logs/`

- [ ] **Step 1: Run smoke training**

Run:

```bash
bash infra/bin/train_lora.sh --gpu 0 --smoke
```

Expected:
- smoke log created
- checkpoints saved in `/mnt/kai_ckp/model/Assignment2_ca6114/ukiyoe/`

- [ ] **Step 2: Review smoke output before full training**

Check:
- log contains no tokenizer / xformers / network errors
- smoke checkpoint exists

- [ ] **Step 3: Run full training**

Run:

```bash
bash infra/bin/train_lora.sh --gpu 0
```

Expected:
- 1200-step run
- multiple checkpoints
- final `ukiyoe_sdxl_style_lora.safetensors`

- [ ] **Step 4: Select the checkpoint to discuss**

Pick one:
- final checkpoint
- or a mid-step checkpoint if visually stronger

Write a short note:
- why this checkpoint was chosen
- what got better
- what artifacts remained

---

### Task 6: Run Automated Validation

**Files:**
- Create: `runtime_ukiyoe/workspace/report_assets/validation_manifest.csv`
- Create: `runtime_ukiyoe/workspace/report_assets/validation_execution.csv`
- Output images under shared ComfyUI output tree

- [ ] **Step 1: Run automated validation**

Run:

```bash
bash infra/bin/run_validation_matrix.sh
```

Expected:
- base model published
- latest LoRA published
- ComfyUI auto-starts if needed
- 72 prompts execute
- execution CSV written

- [ ] **Step 2: Verify execution completeness**

Check:
- `validation_execution.csv` has `72` rows
- all rows `status=completed`
- all `image_files` non-empty

- [ ] **Step 3: Pull out comparison subsets for the report**

Minimum subsets:
- baseline vs LoRA
- simple vs medium vs complex
- best example
- worst example / failure case

---

### Task 7: Build Report B So It Is Genuinely Distinct

**Files:**
- Use: `runtime_ukiyoe/workspace/report_assets/...`
- Output: your final PDF outside infra as needed

- [ ] **Step 1: Change the analytical emphasis**

For Ukiyo-e, focus on:
- contour lines
- flat color regions
- negative space
- print-like composition
- stylization over realism

Do **not** reuse Van Gogh’s emphasis on:
- impasto texture
- turbulent brushstrokes
- thick oil-paint feeling

- [ ] **Step 2: Use a different figure set**

Do not reuse the same prompt/image pair structure as Report A.

Recommended:
- Report A: more portrait/interior emphasis
- Report B: more landscape/travel-scene emphasis

- [ ] **Step 3: Use a different failure analysis**

For Ukiyo-e, good failure themes:
- over-realistic shading breaking print style
- linework inconsistency
- western-perspective drift
- palette too rich / too photographic

- [ ] **Step 4: Change the report structure enough**

Recommended Report B flow:
1. Style target and historical visual traits
2. Dataset design and caption strategy
3. Training setup and checkpoint selection
4. Baseline vs LoRA comparison
5. Prompt complexity comparison
6. Failure modes and future improvement

---

### Task 8: Keep Similarity Low Without Cheating

**Files:**
- Reuse evidence folders only as independent experiment records

- [ ] **Step 1: Keep all experimental assets separate**

Must differ:
- dataset
- trigger token
- checkpoint directory
- validation CSV
- generated outputs

- [ ] **Step 2: Do not recycle wording paragraph-by-paragraph**

Allowed to reuse:
- methodology scaffold
- infrastructure description

Should rewrite:
- style description
- interpretation
- findings
- limitations

- [ ] **Step 3: Use different representative prompts in the written discussion**

Even if the validation matrix is structurally the same, highlight different rows in the report body.

---

### Task 9: Final Packaging

**Files:**
- Keep both experiment evidence trees

- [ ] **Step 1: Archive the Van Gogh evidence**

Do not modify:
- `/mnt/kai_ckp/model/Assignment2_ca6114/vangogh/`
- current Van Gogh report assets

- [ ] **Step 2: Archive the Ukiyo-e evidence**

Keep:
- config
- logs
- chosen checkpoint note
- validation CSVs
- selected comparison images

- [ ] **Step 3: Final pre-submission checklist**

Confirm Report B includes:
- Scenario 3 title
- your own trained LoRA
- prompt complexity comparison
- base vs LoRA comparison
- limitations and future work
- appendix images if needed

---

## Recommendation

If your goal is **two truly independent Scenario 3 reports**, then **yes, train a second LoRA**.  
If your goal is only **one report plus one backup draft**, then you could reuse Van Gogh, but that is not the best use of your current infrastructure.

For your current repo and server state, the highest-value path is:

1. Create `runtime_ukiyoe`
2. Train a Ukiyo-e LoRA
3. Run the same automated validation pipeline
4. Write Report B around linework / flat-color / print-style analysis

