#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', required=True)
    parser.add_argument('--workspace-name', required=True)
    parser.add_argument('--checkpoints-root', required=True)
    parser.add_argument('--loras-root', required=True)
    parser.add_argument('--output-env')

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--lora-path')
    source_group.add_argument('--lora-dir')
    parser.add_argument('--output-prefix')
    return parser.parse_args()


def resolve_latest_lora(lora_dir: Path, output_prefix: str | None) -> Path:
    if output_prefix:
        pattern = f'{output_prefix}*.safetensors'
    else:
        pattern = '*.safetensors'
    candidates = sorted(
        lora_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f'no LoRA checkpoints found in {lora_dir} using pattern {pattern}')
    return candidates[0]


def ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(f'cannot replace non-symlink path: {link_path}')
    link_path.symlink_to(target)


def write_env_file(output_path: Path, values: dict[str, str]) -> None:
    lines = [f"{key}={shell_quote(value)}" for key, value in values.items()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    args = parse_args()
    base_model_path = Path(args.base_model_path).resolve()
    if not base_model_path.is_file():
        raise FileNotFoundError(f'base model missing: {base_model_path}')

    if args.lora_path:
        lora_path = Path(args.lora_path).resolve()
    else:
        lora_path = resolve_latest_lora(Path(args.lora_dir).resolve(), args.output_prefix)
    if not lora_path.is_file():
        raise FileNotFoundError(f'LoRA checkpoint missing: {lora_path}')

    checkpoints_root = Path(args.checkpoints_root).resolve()
    loras_root = Path(args.loras_root).resolve()

    base_model_name = base_model_path.name
    lora_name = f'{args.workspace_name}_latest{lora_path.suffix or ".safetensors"}'

    ensure_symlink(base_model_path, checkpoints_root / base_model_name)
    ensure_symlink(lora_path, loras_root / lora_name)

    values = {
        'BASE_MODEL_NAME': base_model_name,
        'BASE_MODEL_PATH': str(base_model_path),
        'LORA_NAME': lora_name,
        'LORA_PATH': str(lora_path),
    }

    if args.output_env:
        write_env_file(Path(args.output_env), values)

    for key, value in values.items():
        print(f'{key}={value}')


if __name__ == '__main__':
    main()
