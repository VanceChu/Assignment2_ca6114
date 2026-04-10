#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--trigger-token', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--student-id', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(Path(args.template).read_text(encoding='utf-8'))
    rows = []
    for tier in ('simple', 'medium', 'complex'):
        for prompt in data[tier]:
            prompt_text = prompt.replace('<trigger_token>', args.trigger_token)
            for seed in data['seeds']:
                for weight in data['lora_weights']:
                    rows.append({
                        'student_id': args.student_id,
                        'tier': tier,
                        'seed': seed,
                        'lora_weight': weight,
                        'prompt': prompt_text,
                        'negative_prompt': data['negative_prompt'],
                    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['student_id', 'tier', 'seed', 'lora_weight', 'prompt', 'negative_prompt'])
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    print(len(rows))


if __name__ == '__main__':
    main()
