#!/usr/bin/env python3
"""Generate appendix figures: side-by-side comparison pairs and labeled failure cases."""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CELL_SIZE = 512
LABEL_HEIGHT = 40
FONT_SIZE = 20
DPI = 300


def _get_font(size):
    """Try common system fonts, fall back to default."""
    for name in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_pair(left_path, right_path, left_label, right_label, output_path):
    """Create a side-by-side comparison image with labels."""
    left = Image.open(left_path).resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
    right = Image.open(right_path).resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)

    width = CELL_SIZE * 2 + 4  # 4px gap
    height = CELL_SIZE + LABEL_HEIGHT
    canvas = Image.new("RGB", (width, height), "white")

    font = _get_font(FONT_SIZE)
    draw = ImageDraw.Draw(canvas)

    # Draw labels
    for i, label in enumerate([left_label, right_label]):
        x_offset = i * (CELL_SIZE + 4)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x_offset + (CELL_SIZE - tw) // 2
        draw.text((tx, 8), label, fill="black", font=font)

    # Paste images
    canvas.paste(left, (0, LABEL_HEIGHT))
    canvas.paste(right, (CELL_SIZE + 4, LABEL_HEIGHT))

    canvas.save(output_path, dpi=(DPI, DPI))
    print(f"  saved: {output_path}")


def make_single(img_path, label, output_path):
    """Create a labeled single image (for failure cases)."""
    img = Image.open(img_path).resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)

    width = CELL_SIZE
    height = CELL_SIZE + LABEL_HEIGHT
    canvas = Image.new("RGB", (width, height), "white")

    font = _get_font(FONT_SIZE)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = (CELL_SIZE - tw) // 2
    draw.text((tx, 8), label, fill="black", font=font)

    canvas.paste(img, (0, LABEL_HEIGHT))
    canvas.save(output_path, dpi=(DPI, DPI))
    print(f"  saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate appendix figures from a spec file.")
    parser.add_argument("spec", help="JSON spec file with image selections")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    base = Path(spec["image_base"])
    out_dir = Path(spec["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures in {out_dir}")

    for pair in spec.get("pairs", []):
        make_pair(
            base / pair["baseline"],
            base / pair["lora"],
            pair.get("left_label", "Baseline"),
            pair.get("right_label", f"LoRA w={pair.get('weight', '?')}"),
            out_dir / pair["output"],
        )

    for fail in spec.get("failures", []):
        make_single(
            base / fail["image"],
            fail.get("label", "Failure Case"),
            out_dir / fail["output"],
        )

    print(f"Done. {len(spec.get('pairs', []))} pairs + {len(spec.get('failures', []))} failures.")


if __name__ == "__main__":
    main()
