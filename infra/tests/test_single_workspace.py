from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeComfyUIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _FakeComfyUIRequestHandler)
        self.prompt_requests: list[dict[str, object]] = []
        self.history_by_prompt_id: dict[str, dict[str, object]] = {}
        self._next_prompt_index = 1

    def next_prompt_id(self) -> str:
        prompt_id = f'prompt-{self._next_prompt_index}'
        self._next_prompt_index += 1
        return prompt_id


class _FakeComfyUIRequestHandler(BaseHTTPRequestHandler):
    server: _FakeComfyUIServer

    def do_POST(self) -> None:  # noqa: N802
        if self.path != '/prompt':
            self.send_error(404)
            return
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode('utf-8'))
        prompt_id = self.server.next_prompt_id()
        self.server.prompt_requests.append(payload)
        save_image_node = payload['prompt']['10']['inputs']
        self.server.history_by_prompt_id[prompt_id] = {
            prompt_id: {
                'outputs': {
                    '10': {
                        'images': [
                            {
                                'filename': f"{save_image_node['filename_prefix']}.png",
                                'subfolder': 'validation',
                                'type': 'output',
                            }
                        ]
                    }
                }
            }
        }
        response = json.dumps({'prompt_id': prompt_id}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith('/history/'):
            prompt_id = self.path.rsplit('/', 1)[-1]
            payload = self.server.history_by_prompt_id.get(prompt_id, {})
            response = json.dumps(payload).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return


class SingleWorkspaceInfraTests(unittest.TestCase):
    def test_create_workspace_defaults_to_runtime_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env['SCENARIO3_RUNTIME_ROOT'] = tmpdir
            subprocess.run(
                ['bash', 'infra/bin/create_workspace.sh'],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            workspace_root = Path(tmpdir) / 'workspace'
            expected_dirs = [
                'dataset_raw',
                'dataset_curated',
                'captions',
                'dataset_train',
                'configs',
                'checkpoints',
                'outputs',
                'report_assets',
                'logs',
            ]
            for rel in expected_dirs:
                self.assertTrue((workspace_root / rel).is_dir(), rel)

    def test_template_uses_single_workspace_defaults(self) -> None:
        template = (REPO_ROOT / 'infra/templates/sdxl_style_lora.env.example').read_text(encoding='utf-8')
        self.assertIn('WORKSPACE_NAME=workspace', template)
        self.assertNotIn('STUDENT_ID=', template)
        self.assertNotIn('student_a', template)

    def test_bootstrap_host_installs_comfyui_by_default(self) -> None:
        script = (REPO_ROOT / 'infra/bin/bootstrap_host.sh').read_text(encoding='utf-8')
        self.assertIn('INSTALL_COMFYUI="${INSTALL_COMFYUI:-1}"', script)

    def test_bootstrap_host_registers_comfyui_example_workflow(self) -> None:
        script = (REPO_ROOT / 'infra/bin/bootstrap_host.sh').read_text(encoding='utf-8')
        common = (REPO_ROOT / 'infra/bin/common.sh').read_text(encoding='utf-8')
        self.assertIn('COMFYUI_TEMPLATE_NODE_NAME="assignment2_ca6114_templates"', common)
        self.assertIn('COMFYUI_TEMPLATE_WORKFLOWS_ROOT', common)
        self.assertIn('install_comfyui_example_workflow', script)
        self.assertIn('sdxl_style_lora_inference.json', script)

    def test_repo_keeps_manual_comfyui_workflow_template(self) -> None:
        workflow = REPO_ROOT / 'infra/workflows/sdxl_style_lora_inference.json'
        self.assertTrue(workflow.is_file())
        payload = json.loads(workflow.read_text(encoding='utf-8'))
        self.assertIn('3', payload)
        self.assertIn('10', payload)
        self.assertEqual(payload['3']['class_type'], 'CheckpointLoaderSimple')
        self.assertEqual(payload['10']['class_type'], 'SaveImage')

    def test_publish_comfyui_assets_creates_workspace_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_model = tmp_path / 'models' / 'sd_xl_base_1.0.safetensors'
            lora_file = tmp_path / 'checkpoints' / 'workspace_sdxl_style_lora_step1200.safetensors'
            checkpoints_root = tmp_path / 'shared_models' / 'checkpoints'
            loras_root = tmp_path / 'shared_models' / 'loras'
            output_env = tmp_path / 'asset_names.env'
            base_model.parent.mkdir(parents=True, exist_ok=True)
            lora_file.parent.mkdir(parents=True, exist_ok=True)
            base_model.write_text('base', encoding='utf-8')
            lora_file.write_text('lora', encoding='utf-8')

            subprocess.run(
                [
                    'python3',
                    'infra/bin/publish_comfyui_assets.py',
                    '--base-model-path',
                    str(base_model),
                    '--lora-path',
                    str(lora_file),
                    '--checkpoints-root',
                    str(checkpoints_root),
                    '--loras-root',
                    str(loras_root),
                    '--workspace-name',
                    'workspace',
                    '--output-env',
                    str(output_env),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            published_base = checkpoints_root / base_model.name
            published_lora = loras_root / 'workspace_latest.safetensors'
            self.assertTrue(published_base.is_symlink())
            self.assertEqual(published_base.resolve(), base_model.resolve())
            self.assertTrue(published_lora.is_symlink())
            self.assertEqual(published_lora.resolve(), lora_file.resolve())
            env_text = output_env.read_text(encoding='utf-8')
            self.assertIn("BASE_MODEL_NAME='sd_xl_base_1.0.safetensors'", env_text)
            self.assertIn("LORA_NAME='workspace_latest.safetensors'", env_text)

    def test_run_comfyui_batch_submits_rendered_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / 'validation_manifest.csv'
            report_path = tmp_path / 'validation_execution.csv'
            with manifest_path.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        'workspace_name',
                        'tier',
                        'seed',
                        'lora_weight',
                        'prompt',
                        'negative_prompt',
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        'workspace_name': 'workspace',
                        'tier': 'simple',
                        'seed': '101',
                        'lora_weight': 'baseline',
                        'prompt': '<my_style>, coffee mug on a desk',
                        'negative_prompt': 'low quality',
                    }
                )
                writer.writerow(
                    {
                        'workspace_name': 'workspace',
                        'tier': 'complex',
                        'seed': '202',
                        'lora_weight': '0.8',
                        'prompt': '<my_style>, neon city skyline',
                        'negative_prompt': 'blurry',
                    }
                )

            server = _FakeComfyUIServer(('127.0.0.1', 0))
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            try:
                subprocess.run(
                    [
                        'python3',
                        'infra/bin/run_comfyui_batch.py',
                        '--manifest',
                        str(manifest_path),
                        '--workflow-template',
                        'infra/workflows/sdxl_style_lora_inference.json',
                        '--server-url',
                        f'http://127.0.0.1:{server.server_port}',
                        '--base-model-name',
                        'sd_xl_base_1.0.safetensors',
                        '--lora-name',
                        'workspace_latest.safetensors',
                        '--output-report',
                        str(report_path),
                    ],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            finally:
                server.shutdown()
                thread.join()
                server.server_close()

            self.assertEqual(len(server.prompt_requests), 2)
            baseline_prompt = server.prompt_requests[0]['prompt']
            weighted_prompt = server.prompt_requests[1]['prompt']

            self.assertEqual(baseline_prompt['3']['inputs']['ckpt_name'], 'sd_xl_base_1.0.safetensors')
            self.assertEqual(baseline_prompt['4']['inputs']['lora_name'], 'workspace_latest.safetensors')
            self.assertEqual(baseline_prompt['4']['inputs']['strength_model'], 0.0)
            self.assertEqual(baseline_prompt['4']['inputs']['strength_clip'], 0.0)
            self.assertEqual(baseline_prompt['5']['inputs']['text'], '<my_style>, coffee mug on a desk')
            self.assertEqual(baseline_prompt['6']['inputs']['text'], 'low quality')
            self.assertEqual(baseline_prompt['8']['inputs']['seed'], 101)
            self.assertIn('baseline', baseline_prompt['10']['inputs']['filename_prefix'])

            self.assertEqual(weighted_prompt['4']['inputs']['strength_model'], 0.8)
            self.assertEqual(weighted_prompt['4']['inputs']['strength_clip'], 0.8)
            self.assertEqual(weighted_prompt['5']['inputs']['text'], '<my_style>, neon city skyline')
            self.assertEqual(weighted_prompt['6']['inputs']['text'], 'blurry')
            self.assertEqual(weighted_prompt['8']['inputs']['seed'], 202)
            self.assertIn('w0p8', weighted_prompt['10']['inputs']['filename_prefix'])

            with report_path.open(newline='', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['status'], 'completed')
            self.assertTrue(rows[0]['image_files'].endswith('.png'))


if __name__ == '__main__':
    unittest.main()
