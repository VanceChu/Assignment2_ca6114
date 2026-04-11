#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--workflow-template', required=True)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--base-model-name', required=True)
    parser.add_argument('--lora-name', required=True)
    parser.add_argument('--output-report', required=True)
    parser.add_argument('--timeout-seconds', type=float, default=300.0)
    parser.add_argument('--poll-interval', type=float, default=1.0)
    return parser.parse_args()


def normalize_server_url(server_url: str) -> str:
    return server_url.rstrip('/')


def parse_lora_weight(raw_weight: str) -> float:
    if raw_weight == 'baseline':
        return 0.0
    return float(raw_weight)


def weight_tag(raw_weight: str) -> str:
    if raw_weight == 'baseline':
        return 'baseline'
    return f"w{raw_weight.replace('.', 'p')}"


def prompt_tag(prompt_text: str) -> str:
    cleaned = re.sub(r'[^a-z0-9]+', '_', prompt_text.lower()).strip('_')
    if not cleaned:
        return 'prompt'
    return cleaned[:40]


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def render_workflow(
    template: dict[str, object],
    row: dict[str, str],
    row_index: int,
    base_model_name: str,
    lora_name: str,
) -> dict[str, object]:
    workflow = copy.deepcopy(template)
    strength = parse_lora_weight(row['lora_weight'])
    prefix = (
        f"validation/{row['workspace_name']}/{row['tier']}/"
        f"r{row_index:03d}_{weight_tag(row['lora_weight'])}_"
        f"seed{row['seed']}_{prompt_tag(row['prompt'])}"
    )

    workflow['3']['inputs']['ckpt_name'] = base_model_name
    workflow['4']['inputs']['lora_name'] = lora_name
    workflow['4']['inputs']['strength_model'] = strength
    workflow['4']['inputs']['strength_clip'] = strength
    workflow['5']['inputs']['text'] = row['prompt']
    workflow['6']['inputs']['text'] = row['negative_prompt']
    workflow['8']['inputs']['seed'] = int(row['seed'])
    workflow['10']['inputs']['filename_prefix'] = prefix
    return workflow


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode('utf-8'))


def get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode('utf-8'))


def queue_prompt(server_url: str, prompt: dict[str, object]) -> str:
    response = post_json(
        f'{server_url}/prompt',
        {
            'prompt': prompt,
            'client_id': 'assignment2_ca6114_infra',
        },
    )
    prompt_id = response.get('prompt_id')
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError(f'invalid ComfyUI queue response: {response}')
    return prompt_id


def wait_for_prompt_completion(
    server_url: str,
    prompt_id: str,
    timeout_seconds: float,
    poll_interval: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    history_url = f'{server_url}/history/{prompt_id}'
    while time.monotonic() < deadline:
        try:
            history = get_json(history_url)
        except urllib.error.URLError:
            time.sleep(poll_interval)
            continue
        prompt_history = history.get(prompt_id)
        if isinstance(prompt_history, dict) and prompt_history.get('outputs'):
            return prompt_history
        time.sleep(poll_interval)
    raise TimeoutError(f'timed out waiting for ComfyUI prompt {prompt_id}')


def collect_images(prompt_history: dict[str, object]) -> list[str]:
    outputs = prompt_history.get('outputs', {})
    if not isinstance(outputs, dict):
        return []
    image_paths: list[str] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get('images', [])
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            filename = image.get('filename', '')
            subfolder = image.get('subfolder', '')
            if not filename:
                continue
            if subfolder:
                image_paths.append(f'{subfolder}/{filename}')
            else:
                image_paths.append(str(filename))
    return image_paths


def write_report(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                'workspace_name',
                'tier',
                'seed',
                'lora_weight',
                'prompt',
                'negative_prompt',
                'prompt_id',
                'status',
                'image_files',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest_rows(Path(args.manifest))
    workflow_template = json.loads(Path(args.workflow_template).read_text(encoding='utf-8'))
    server_url = normalize_server_url(args.server_url)

    report_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(manifest_rows):
        prompt = render_workflow(
            workflow_template,
            row,
            row_index=row_index,
            base_model_name=args.base_model_name,
            lora_name=args.lora_name,
        )
        prompt_id = queue_prompt(server_url, prompt)
        prompt_history = wait_for_prompt_completion(
            server_url,
            prompt_id,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
        )
        report_rows.append(
            {
                **row,
                'prompt_id': prompt_id,
                'status': 'completed',
                'image_files': ';'.join(collect_images(prompt_history)),
            }
        )

    write_report(Path(args.output_report), report_rows)
    print(Path(args.output_report))
    print(len(report_rows))


if __name__ == '__main__':
    main()
