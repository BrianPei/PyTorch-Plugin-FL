# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise MUSA environment reuse without hardware, downloads or package installs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/set_env_musa.sh"
).read_text()
REVISION = "437ba39387ddc681dc884259ef9dbf0c1802bccc"
REPOSITORY = "https://github.com/flagos-ai/FlagGems.git"


def shell_function(name):
    start = SCRIPT.index(f"{name}() {{")
    end = SCRIPT.index("\n}\n", start) + 3
    return SCRIPT[start:end]


def run_shell(body, **environment):
    env = os.environ.copy()
    env.update(PYTHONPATH="", PYTHONNOUSERSITE="1")
    env.update(environment)
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + body],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "condition,usable",
    [
        ("isolated", True),
        ("system_packages", False),
        ("vendor_module", False),
        ("orphan_vendor_metadata", False),
        ("missing_pip", False),
    ],
)
def test_venv_reuse_requires_isolation(tmp_path, condition, usable):
    root = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
    )
    site = next((root / "lib").glob("python*/site-packages"))
    if condition != "missing_pip":
        (site / "pip.py").touch()
    if condition == "system_packages":
        config = root / "pyvenv.cfg"
        config.write_text(config.read_text().replace("= false", "= true"))
    if condition == "vendor_module":
        (site / "torch_musa.py").touch()
    if condition == "orphan_vendor_metadata":
        dist = site / "torch_musa-2.9.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Name: torch_musa\nVersion: 2.9.1\n")

    result = run_shell(
        shell_function("venv_is_usable") + "\nvenv_is_usable",
        VENV_ROOT=str(root),
    )
    assert (result.returncode == 0) == usable, result.stdout + result.stderr


@pytest.mark.parametrize(
    "condition,reuse",
    [
        ("same_commit", True),
        ("different_commit", False),
        ("different_repo", False),
        ("missing_module", False),
        ("missing_provenance", False),
        ("broken_provenance", False),
        ("branch", False),
    ],
)
def test_flaggems_reuse_requires_exact_source(tmp_path, condition, reuse):
    dist = tmp_path / "flag_gems-5.4.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: flag_gems\nVersion: 5.4.0\n")
    if condition != "missing_module":
        (tmp_path / "flag_gems.py").touch()
    origin = {
        "url": REPOSITORY,
        "vcs_info": {"vcs": "git", "commit_id": REVISION},
    }
    if condition == "different_commit":
        origin["vcs_info"]["commit_id"] = "a" * 40
    if condition == "different_repo":
        origin["url"] = "https://example.invalid/FlagGems.git"
    if condition != "missing_provenance":
        (dist / "direct_url.json").write_text(
            "invalid" if condition == "broken_provenance" else json.dumps(origin)
        )

    result = run_shell(
        "pip_retry() { echo INSTALL_REQUESTED; }\n"
        "flag_gems_installed() { return 0; }\n"
        + shell_function("install_flag_gems")
        + "\ninstall_flag_gems",
        VENV_PYTHON=sys.executable,
        PYTHONPATH=str(tmp_path),
        FLAGGEMS_REPO=REPOSITORY,
        FLAGGEMS_REVISION="master" if condition == "branch" else REVISION,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("INSTALL_REQUESTED" not in result.stdout) == reuse


@pytest.mark.parametrize("qwen", ["0", "1"])
def test_integration_keeps_bert_and_gates_tokenizer_deps(tmp_path, qwen):
    # Execute the dependency block with an offline recording Python command.
    recorder = tmp_path / "python"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    recorder.chmod(0o755)
    start = SCRIPT.index('if [[ "$CI_STAGE" == "integration" ]]; then')
    end = SCRIPT.index("\nexport VIRTUAL_ENV=", start)
    result = run_shell(
        SCRIPT[start:end],
        CI_STAGE="integration",
        VENV_PYTHON=str(recorder),
        PIP_INDEX_URL_ARG="https://example.invalid/simple",
        TORCH_FL_INSTALL_QWEN_DEPS=qwen,
    )
    assert result.returncode == 0, result.stderr
    assert "transformers>=4.51,<5" in result.stdout
    assert "PyYAML==6.0.1" in result.stdout
    for package in ("sentencepiece", "tiktoken", "protobuf"):
        assert (package in result.stdout) == (qwen == "1")
