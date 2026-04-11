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

## External Checkpoint Root

```text
/mnt/kai_ckp/model/Assignment2_ca6114
```

Each workspace writes LoRA checkpoints into:

```text
/mnt/kai_ckp/model/Assignment2_ca6114/<workspace_name>/
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
- LoRA checkpoint outputs are stored outside `runtime/` under `/mnt/kai_ckp/model/Assignment2_ca6114/<workspace_name>/`.
- `runtime/workspace/` is the only place you need for the single-user pipeline.
- GPU locks live under `runtime/shared/locks/gpu<id>.lock`.
