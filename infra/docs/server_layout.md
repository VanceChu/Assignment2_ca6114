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
  workspace/
    dataset_raw/
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

- `runtime/shared/` stores shared code, locks, and model assets.
- `runtime/workspace/` is the only place you need for the single-user pipeline.
- GPU locks live under `runtime/shared/locks/gpu<id>.lock`.
