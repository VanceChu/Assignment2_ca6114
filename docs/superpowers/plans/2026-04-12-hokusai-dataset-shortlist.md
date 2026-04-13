# Hokusai Dataset Shortlist

## Runtime

Second experiment runtime:

```text
/mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime_hokusai
```

Shared symlinks:

```text
runtime_hokusai/shared -> /mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime/shared
runtime_hokusai/venvs  -> /mnt/world_foundational_model/kai/chuzhong/Assignment2_ca6114/runtime/venvs
```

Config:

```dotenv
WORKSPACE_NAME=hokusai
TRIGGER_TOKEN='<hokusai_style>'
OUTPUT_PREFIX=hokusai_sdxl_style_lora
BASE_MODEL=/mnt/kai_ckp/model/Assignment2_ca6114/sd_xl_base_1.0.safetensors
```

## Caption Strategy

Use `A`, but not free-form chaos.

Structure:

```text
<hokusai_style>, ukiyo-e woodblock print, [scene-specific subject], flat color planes, indigo-heavy palette, bold contour lines, Edo-period Japanese print composition
```

Rules:

- Keep the same style spine across all images
- Change the scene phrase per image
- Prefer landscape / water / bridge / travel-scene wording
- Avoid repeating the artwork title verbatim unless it helps identify the scene
- Avoid generic filler like "beautiful artwork"

## Selection Principles

Goal: maximize style learning, minimize content overfitting.

Keep:
- color woodblock prints
- landscapes
- rivers
- bridges
- boats
- waterfalls
- travel scenes
- mountain / Fuji views in moderation

Avoid:
- portraits
- actor prints
- ghosts / dramatic character art
- monochrome sketches
- flower-and-bird clusters if they dominate the set
- too many nearly identical Fuji-only compositions

## Primary Set

Recommended primary set size: `24`

### Water / Coast / Boat

1. `cargo-ship-and-wave`
2. `the-great-wave-of-kanagawa-1831`
3. `shore-of-tago-bay-ejiri-at-tokaido`
4. `bay-of-noboto`
5. `fishing-in-the-bay-uraga`
6. `the-kazusa-sea-route`

### River / Bridge / Townscape

7. `view-of-honmoku`
8. `the-fields-of-sekiya-by-the-sumida-river-1831`
9. `the-festival-of-lanterns-on-temma-bridge-1834`
10. `the-pontoon-bridge-at-sano-in-the-province-of-kozuka`
11. `fishing-in-the-river-kinu`
12. `moonlight-over-the-sumida-river-in-edo`
13. `nihonbashi-bridge-in-edo`
14. `sunset-across-the-ryogoku-bridge-from-the-bank-of-the-sumida-river-at-onmagayashi`
15. `tama-river-in-the-musashi-province`

### Fuji / Mountain Landscapes

16. `rainstorm-beneath-the-summit`
17. `fuji-mountains-in-clear-weather-1831`
18. `view-of-fuji-from-a-boat-at-ushibori-1837`
19. `fuji-from-the-platform-of-sasayedo`
20. `fuji-seen-through-the-mannen-bridge-at-fukagawa`
21. `the-lake-of-hakone-in-the-segami-province`

### Waterfalls / Vertical Composition

22. `the-waterfall-of-amida-behind-the-kiso-road`
23. `ono-waterfall-at-kisokaido`
24. `waterfall-at-yoshino-in-washu`

## Backup Set

Use these if any primary image is low resolution, text-heavy, or low quality:

1. `view-on-mount-fuji-between-flowerin-trees`
2. `the-river-tone-in-the-province-of-kazusa`
3. `bridge-in-the-clouds`
4. `lake-suwa-in-the-shinano-province`
5. `fishing-by-torchlight-in-kai-province-from-oceans-of-wisdom-1833`
6. `the-suspension-bridge-between-hida-and-etchu`

## Why This Set Is Better Than A Full Fuji-Only Set

- keeps Hokusai's print language
- increases scene diversity
- avoids teaching the LoRA that "Fuji mountain" is mandatory content
- still preserves strong blue palette and contour-line priors

## Next Step

Download the `24` primary images first.

If fewer than `20` are high enough quality, pull from the backup set until you have `24` good images, then trim to the strongest `20-24`.
