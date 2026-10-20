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

"""Execute ci.yml's trusted checkout resolution against a local event payload.

The step under test is the whole of `resolve-checkout`'s decision, run from a
directory that holds only a `control/` checkout: the resolver, the contract it
validates against, and a fake `gh` that answers exactly the paths a dispatch
needs. A pull request is expected to resolve without reaching `gh` at all --
the payload already carries the immutable head and base, and the fake fails
the test if it is called with an unexpected path.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
JOBS = WORKFLOW["jobs"]
SHA = "a" * 40
HEAD = "b" * 40
BASE = "c" * 40
CONTROL = "d" * 40


def _resolve_step():
    steps = [step for step in JOBS["resolve-checkout"]["steps"] if step.get("id") == "resolve"]
    assert len(steps) == 1
    return steps[0]


RUN = _resolve_step()["run"]


def _resolve(tmp_path, event, event_name="pull_request", gh_responses=None):
    control = tmp_path / "control"
    shutil.copytree(ROOT / ".github/scripts", control / ".github/scripts")
    shutil.copytree(ROOT / ".flagcicd", control / ".flagcicd")
    (tmp_path / "event.json").write_text(json.dumps(event))

    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "responses = json.loads(os.environ['FAKE_GH'])\n"
        "path = sys.argv[-1]\n"
        "if path not in responses:\n"
        "    raise SystemExit(f'unexpected gh api {path}')\n"
        "print(json.dumps(responses[path]))\n"
    )
    gh.chmod(0o755)
    # The step invokes `python`, which is not guaranteed to exist on a bare
    # macOS runner; the shim keeps the test on the interpreter running pytest,
    # which is the one that has PyYAML installed.
    (binary / "python").symlink_to(sys.executable)

    output = tmp_path / "output"
    output.touch()
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json"),
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_REPOSITORY": "flagos-ai/Torch-FL",
        "GITHUB_SHA": SHA,
        "GITHUB_WORKFLOW_SHA": CONTROL,
        "GITHUB_OUTPUT": str(output),
        "FAKE_GH": json.dumps(gh_responses or {}),
        "DISPATCH_PLATFORMS": "all",
    }
    process = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", RUN],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    fields = {}
    for line in output.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return process, fields


def test_resolution_uses_the_trusted_control_checkout_and_contract():
    checkout = JOBS["resolve-checkout"]["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.workflow_sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert "control/.github/scripts/flagcicd_resolve.py" in RUN
    assert "--contract control/.flagcicd/project.yaml" in RUN


def test_pull_request_resolves_head_and_base_from_the_payload(tmp_path):
    event = {
        "pull_request": {
            "number": 42,
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": HEAD},
            "base": {"repo": {"full_name": "flagos-ai/Torch-FL"}, "sha": BASE},
            "labels": [{"name": "platform:musa"}],
        }
    }
    # No gh responses: reaching the API for a pull request fails the test.
    process, output = _resolve(tmp_path, event)
    assert process.returncode == 0, process.stderr
    assert output["source_repository"] == "contributor/fork"
    assert output["source_sha"] == HEAD
    assert output["base_sha"] == BASE
    assert output["control_sha"] == CONTROL
    assert output["profile"] == "pr"
    assert output["selection"] == "affected"
    assert output["pr_number"] == "42"


@pytest.mark.parametrize(
    "pr",
    [
        {"head": {"repo": {"full_name": "contributor/fork"}, "sha": "main"}},
        {"head": {"repo": None, "sha": HEAD}},
        {"head": {"repo": {"full_name": "bad/repo/extra"}, "sha": HEAD}},
        {"head": {"sha": HEAD}},
    ],
)
def test_a_pull_request_head_that_is_not_immutable_fails(tmp_path, pr):
    process, output = _resolve(tmp_path, {"pull_request": {"number": 42, **pr}})
    assert process.returncode != 0
    assert output == {}


def test_push_resolves_the_pushed_commit_and_its_comparison(tmp_path):
    event = {
        "ref": "refs/heads/main",
        "before": BASE,
        "after": HEAD,
        "repository": {"full_name": "flagos-ai/Torch-FL"},
    }
    process, output = _resolve(tmp_path, event, event_name="push")
    assert process.returncode == 0, process.stderr
    assert output["source_sha"] == HEAD
    assert output["base_sha"] == BASE
    assert output["profile"] == "pr"
    assert output["selection"] == "affected"


def test_a_new_branch_has_no_comparison_commit(tmp_path):
    event = {
        "ref": "refs/heads/new",
        "before": "0" * 40,
        "after": HEAD,
        "repository": {"full_name": "flagos-ai/Torch-FL"},
    }
    process, output = _resolve(tmp_path, event, event_name="push")
    assert process.returncode == 0, process.stderr
    assert output["base_sha"] == ""
    # No diff is not "nothing changed": the plan runs every platform and reads
    # the missing comparison at the contract's conservative level.
    assert output["selection"] == "all"


def test_a_scheduled_run_resolves_the_default_branch_commit(tmp_path):
    process, output = _resolve(tmp_path, {"schedule": "0 18 * * *"}, "schedule")
    assert process.returncode == 0, process.stderr
    assert output["source_sha"] == SHA
    assert output["base_sha"] == ""
    assert output["profile"] == "nightly"
    assert output["selection"] == "all"


def test_every_declared_cron_is_also_scheduled():
    # The resolver maps a cron to a profile by reading `schedules` from the
    # contract, so the two sides must agree: a cron declared but not triggered
    # here never runs (the profile is dead), and a cron triggered but not
    # declared fails resolution by name. PyYAML reads the unquoted `on:` key as
    # the boolean True, hence the fallback lookup.
    contract = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
    # PyYAML 1.1 reads `on:` as True; the quoted form is what most parsers keep.
    triggers = WORKFLOW.get("on") or WORKFLOW.get(True) or {}
    scheduled = {entry["cron"] for entry in triggers.get("schedule") or []}
    declared = set(contract["schedules"])
    assert scheduled == declared, (
        f"ci.yml schedules {sorted(scheduled)}, project.yaml declares "
        f"{sorted(declared)}"
    )


def test_a_weekly_schedule_resolves_the_weekly_profile(tmp_path):
    # The weekly budget is a different case set and level than the daily one;
    # resolving both crons to nightly would make the weekly profile dead.
    contract = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
    weekly = [cron for cron, profile in contract["schedules"].items() if profile == "weekly"]
    assert weekly, "the contract declares no weekly schedule"
    process, output = _resolve(tmp_path, {"schedule": weekly[0]}, "schedule")
    assert process.returncode == 0, process.stderr
    assert output["profile"] == "weekly"


def test_dispatch_by_pr_number_reads_the_pr_through_the_api(tmp_path):
    responses = {
        "repos/flagos-ai/Torch-FL/pulls/42": {
            "state": "open",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": HEAD},
            "base": {"sha": BASE},
        }
    }
    event = {"inputs": {"pr_number": "42", "checkout_ref": "", "base_ref": ""}}
    process, output = _resolve(tmp_path, event, "workflow_dispatch", responses)
    assert process.returncode == 0, process.stderr
    assert output["source_repository"] == "contributor/fork"
    assert output["source_sha"] == HEAD
    assert output["base_sha"] == BASE


@pytest.mark.parametrize("number", ["0", "-1", "01", "42; echo unsafe", "$(whoami)"])
def test_invalid_pr_numbers_fail_before_any_checkout(tmp_path, number):
    # The number is data from the event payload, never interpolated into a
    # shell, so the injection-shaped values are rejected as syntax, not run.
    event = {"inputs": {"pr_number": number}}
    process, output = _resolve(tmp_path, event, "workflow_dispatch")
    assert process.returncode != 0
    assert output == {}


@pytest.mark.parametrize(
    "pr",
    [
        {
            "state": "closed",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": SHA},
            "base": {"sha": BASE},
        },
        {"state": "open", "head": {"repo": None, "sha": SHA}, "base": {"sha": BASE}},
        {
            "state": "open",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": "main"},
            "base": {"sha": BASE},
        },
    ],
)
def test_unavailable_or_invalid_heads_fail_before_checkout(tmp_path, pr):
    event = {"inputs": {"pr_number": "42"}}
    process, output = _resolve(
        tmp_path, event, "workflow_dispatch", {"repos/flagos-ai/Torch-FL/pulls/42": pr}
    )
    assert process.returncode != 0
    assert output == {}


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
