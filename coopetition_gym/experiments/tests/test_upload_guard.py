"""Test the publishing helper with fake clients; no network or upload occurs."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="bash helper")
@pytest.mark.parametrize("observed,expected_success", [("True", True), ("False", False), ("None", False)])
def test_private_upload_requires_verified_private_destination(tmp_path, observed, expected_success):
    source = Path(__file__).resolve().parents[1] / "croissant" / "upload.sh"
    root = tmp_path / "checkout"
    script = root / "coopetition_gym/experiments/croissant/upload.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(source, script)
    (script.parent / "hf_readme_training.md").write_text("reviewed fixture")
    metadata = root / "papers/neurips_ed_2026/croissant.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}")
    payload = tmp_path / "payload"
    (payload / "training_runs").mkdir(parents=True)
    clients = tmp_path / "clients"
    clients.mkdir()
    log = tmp_path / "calls"
    (clients / "hf").write_text('#!/bin/sh\nprintf "%s\n" "$*" >> "$CALL_LOG"\n')
    (clients / "hf").chmod(0o755)
    (clients / "python3").symlink_to(sys.executable)
    (clients / "huggingface_hub.py").write_text(
        'from types import SimpleNamespace\n'
        'class HfApi:\n'
        '    def create_repo(self, **kwargs):\n'
        '        assert kwargs["private"] is True and kwargs["exist_ok"] is True\n'
        '    def dataset_info(self, repo_id):\n'
        f'        return SimpleNamespace(private={observed}, id=repo_id, sha="test", siblings=[])\n')
    env = dict(os.environ, PATH=str(clients)+os.pathsep+os.environ["PATH"],
               PYTHONPATH=str(clients), SOURCE_ROOT=str(payload), CALL_LOG=str(log))
    env.pop("HF_PRIVATE", None)
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
    calls = log.read_text()
    assert (result.returncode == 0) is expected_success, result.stderr
    assert ("upload " in calls) is expected_success
