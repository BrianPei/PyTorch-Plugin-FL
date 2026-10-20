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

"""Execute trusted checkout resolution with a fake, local-only GitHub CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
RUN = next(
    step["run"]
    for step in WORKFLOW["jobs"]["resolve-checkout"]["steps"]
    if step.get("id") == "resolve"
).replace(
    ".github/scripts/flagcicd_resolve.py",
    str(ROOT / ".github/scripts/flagcicd_resolve.py"),
)
SHA = "a" * 40


def _resolve(tmp_path, number="", pr=None):
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,sys\n"
        "path=sys.argv[-1]\n"
        "if '/pulls/' in path: data=json.loads(os.environ['FAKE_PR'])\n"
        "elif '/commits/' in path: data={'sha':os.environ['EVENT_SHA']}\n"
        "else: data={'default_branch':'main'}\n"
        "print(json.dumps(data))\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "output"
    output.touch()
    if pr is not None:
        pr = {"base": {"sha": "c" * 40}, **pr}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"inputs": {"pr_number": number}}))
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "flagos-ai/Torch-FL",
        "GITHUB_SHA": SHA,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_EVENT_PATH": str(event_path),
        "EVENT_SHA": SHA,
        "GITHUB_OUTPUT": str(output),
        "FAKE_PR": json.dumps(pr or {}),
    }
    process = subprocess.run(
        ["bash", "-e", "-c", RUN], env=env, cwd=tmp_path, capture_output=True, text=True
    )
    return process, output.read_text()


def test_normal_run_pins_the_event_commit(tmp_path):
    process, output = _resolve(tmp_path)
    assert process.returncode == 0, process.stderr
    fields = dict(line.split("=", 1) for line in output.splitlines())
    assert fields["repository"] == "flagos-ai/Torch-FL"
    assert fields["ref"] == SHA
    assert json.loads(fields["request"])["selection"] == "all"


def test_fork_run_resolves_both_repository_and_immutable_head(tmp_path):
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "contributor/fork"}, "sha": "b" * 40},
    }
    process, output = _resolve(tmp_path, "42", pr)
    assert process.returncode == 0, process.stderr
    fields = dict(line.split("=", 1) for line in output.splitlines())
    assert fields["repository"] == "contributor/fork"
    assert fields["ref"] == "b" * 40
    request = json.loads(fields["request"])
    assert request["target_kind"] == "head"
    assert request["base_sha"] == "c" * 40


@pytest.mark.parametrize("number", ["0", "-1", "01", "42; echo unsafe", "$(whoami)"])
def test_invalid_pr_numbers_fail_before_any_checkout(tmp_path, number):
    process, output = _resolve(tmp_path, number)
    assert process.returncode == 1
    assert output == ""


@pytest.mark.parametrize(
    "pr",
    [
        {
            "state": "closed",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": SHA},
        },
        {"state": "open", "head": {"repo": None, "sha": SHA}},
        {
            "state": "open",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": "main"},
        },
        {
            "state": "open",
            "head": {"repo": {"full_name": "bad/repo/extra"}, "sha": SHA},
        },
    ],
)
def test_unavailable_or_invalid_heads_fail_before_checkout(tmp_path, pr):
    process, output = _resolve(tmp_path, "42", pr)
    assert process.returncode == 1
    assert output == ""


def test_label_events_and_dispatches_do_not_cancel_each_other_or_main_runs():
    events = WORKFLOW[True]
    assert {"labeled", "unlabeled"} <= set(events["pull_request"]["types"])
    assert "paths-ignore" not in events["push"]
    group = WORKFLOW["concurrency"]["group"]
    assert "github.event_name" in group
    assert "github.event.pull_request.number" in group
    assert "inputs.pr_number" in group


def test_privileged_control_workflows_only_checkout_the_trusted_event_commit():
    for filename in ("slash-command.yml", "flaky-retry.yml"):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
        for job in workflow["jobs"].values():
            checkouts = [
                step
                for step in job["steps"]
                if step.get("uses", "").startswith("actions/checkout@")
            ]
            assert len(checkouts) == 1
            assert checkouts[0]["with"]["ref"] == "${{ github.sha }}"
            assert checkouts[0]["with"]["persist-credentials"] is False
        assert "contents" in workflow["permissions"]
