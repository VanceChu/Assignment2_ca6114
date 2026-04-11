from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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


if __name__ == '__main__':
    unittest.main()
