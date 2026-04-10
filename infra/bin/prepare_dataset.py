#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-root', required=True)
    parser.add_argument('--trigger-token', required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--min-images', type=int, default=20)
    parser.add_argument('--max-images', type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_root = Path(args.student_root)
    dataset_dir = student_root / 'dataset_curated'
    captions_dir = student_root / 'captions'
    train_dir = student_root / 'dataset_train' / f'{args.repeats}_{args.trigger_token}'

    images = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < args.min_images or len(images) > args.max_images:
        raise SystemExit(f'image count must be between {args.min_images} and {args.max_images}, got {len(images)}')

    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        caption_path = captions_dir / f'{image_path.stem}.txt'
        if not caption_path.exists():
            raise SystemExit(f'missing caption: {caption_path}')
        caption = caption_path.read_text(encoding='utf-8').strip()
        if args.trigger_token not in caption:
            raise SystemExit(f'trigger token missing in {caption_path}')
        with Image.open(image_path) as image:
            width, height = image.size
        if min(width, height) < 1024:
            raise SystemExit(f'image too small: {image_path} -> {width}x{height}')
        shutil.copy2(image_path, train_dir / image_path.name)
        shutil.copy2(caption_path, train_dir / caption_path.name)

    print(f'training_data={train_dir}')
    print(f'image_count={len(images)}')


if __name__ == '__main__':
    main()
