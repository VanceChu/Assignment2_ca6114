# Server Layout

## Repo Root

```text
/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114
```

## Runtime Root

```text
<repo>/runtime
```

Resolved on `6-a800` as:

```text
/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime
```

## Runtime Tree

```text
runtime/
  shared/
    comfyui/
    sd-scripts/
    logs/
    locks/
    models/
      checkpoints/
      loras/
      vae/
  students/
    student_a/
      dataset_raw/
      dataset_curated/
      captions/
      dataset_train/
      configs/
      checkpoints/
      outputs/
      report_assets/
      logs/
    student_b/
    student_c/
```

## Ownership Rules

- Shared code and shared models live under `runtime/shared/`.
- Student-specific artifacts live under `runtime/students/<student_id>/`.
- Each GPU lock is stored under `runtime/shared/locks/gpu<id>.lock`.
