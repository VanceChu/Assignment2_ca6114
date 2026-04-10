# Student Workflow

## 1. Copy the config template

```bash
cp infra/templates/sdxl_style_lora.env.example runtime/students/<student_id>/configs/sdxl_style_lora.env
```

Update:

- `STUDENT_ID`
- `TRIGGER_TOKEN`
- `BASE_MODEL`
- optional train hyperparameters

## 2. Prepare your dataset

- Put `20-30` curated style images into `dataset_curated/`.
- Put one `.txt` caption per image into `captions/`.
- Every caption must include the trigger token.
- All images should represent one coherent visual style.

## 3. Validate and build the train folder

```bash
bash infra/bin/prepare_dataset.sh <student_id>
```

This validates image count, caption parity, trigger token usage, image size, and creates `dataset_train/10_<trigger_token>/` for `sd-scripts`.

## 4. Run LoRA training

```bash
bash infra/bin/train_student_lora.sh <student_id> --gpu 0
```

For a short smoke run:

```bash
bash infra/bin/train_student_lora.sh <student_id> --gpu 0 --smoke
```

## 5. Keep report evidence

Retain these files:

- dataset sheet or folder screenshot
- three sample captions
- final config file
- training logs
- saved checkpoints
- generated validation grid or manifest
- one failure case with notes
