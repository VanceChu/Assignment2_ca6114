# Single Workspace Workflow

## 1. Initialize the workspace

```bash
bash infra/bin/create_workspace.sh
cp infra/templates/sdxl_style_lora.env.example runtime/workspace/configs/sdxl_style_lora.env
```

Edit the config and set:

- `TRIGGER_TOKEN`
- `BASE_MODEL`
- optional train hyperparameters

## 2. Prepare your dataset

Put your data into:

- `runtime/workspace/dataset_curated/`
- `runtime/workspace/captions/`

Rules:

- use `20-30` curated images
- provide one same-name `.txt` caption per image
- every caption must include the trigger token
- keep the style coherent across the full dataset

## 3. Validate and build the train folder

```bash
bash infra/bin/prepare_dataset.sh
```

This validates image count, caption parity, trigger token usage, image size, and builds `runtime/workspace/dataset_train/`.

## 4. Run LoRA training

Smoke test:

```bash
bash infra/bin/train_lora.sh --gpu 0 --smoke
```

Full run:

```bash
bash infra/bin/train_lora.sh --gpu 0
```

Checkpoints are saved under `/mnt/kai_ckp/model/Assignment2_ca6114/<workspace_name>/`.

## 5. Generate the validation manifest

```bash
bash infra/bin/run_validation_matrix.sh
```

This writes:

- `runtime/workspace/report_assets/validation_manifest.csv`
- `runtime/workspace/report_assets/validation_execution.csv`

It also publishes your latest LoRA into the shared ComfyUI model path and queues the validation batch through ComfyUI automatically.

## 6. Keep report evidence

Retain these files:

- dataset sheet or folder screenshot
- three sample captions
- final config file
- training logs
- saved checkpoints
- generated validation grid or manifest
- one failure case with notes
