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

"""Compare observed plans with the existing router and loader command boundary."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "observed_plan", SCRIPTS / "flagcicd_plan.py"
    )
    PLANNER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PLANNER)
finally:
    sys.path.pop(0)
CONTRACT = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())


def request(**updates):
    return {
        "source_repository": "owner/fork",
        "source_sha": "a" * 40,
        "control_sha": "c" * 40,
        "base_sha": "b" * 40,
        "requested_level": "auto",
        "platforms": "all",
        "task_id": "",
        "parent_run_id": "",
        **updates,
    }


@pytest.mark.parametrize("profile", ["pr", "nightly"])
def test_every_planned_lane_matches_the_existing_loader_cli(tmp_path, profile):
    plan = PLANNER.build_plan(request(profile=profile), CONTRACT, ROOT, None)
    assert [lane["name"] for lane in plan["execution_plan"]] == plan[
        "expected_platforms"
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            yaml.safe_load((ROOT / ".github/integration-manifest.yml").read_text())
        )
    )
    config_file = tmp_path / "config.json"
    for lane in plan["execution_plan"]:
        config = yaml.safe_load(
            (ROOT / f".github/configs/{lane['platform']}.yml").read_text()
        )
        boxing = lane["lane"] == "boxing"
        if boxing:
            config["display_name"] += " (CUDA boxing)"
        config_file.write_text(json.dumps(config))
        # These are the exact CLI inputs supplied by load-platform-config.yml.
        command = [
            sys.executable,
            str(SCRIPTS / "compose_integration_manifest.py"),
            "--platform",
            lane["platform"],
            "--profile",
            profile,
            "--manifest",
            str(manifest),
            "--platform-config",
            str(config_file),
            "--config-dir",
            str(ROOT / ".github/configs"),
            "--environment-check",
            json.dumps(
                config.get(
                    "boxing_environment_check" if boxing else "environment_check"
                )
            ),
        ]
        if boxing:
            command.append("--boxing-lane")
        expected = json.loads(
            subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
        )
        assert [
            {k: v for k, v in g.items() if k != "group_id"} for g in lane["groups"]
        ] == expected
        ids = [group["group_id"] for group in lane["groups"]]
        assert ids == [group["manifest_id"] for group in expected]
        assert len(ids) == len(set(ids))
        assert lane["environment"] == config.get("integration_environment", {})


@pytest.mark.parametrize(
    "event,changed,labels,dispatch,reason",
    [
        ("pull_request", ["README.md"], [], "", "documentation_or_empty_diff"),
        ("pull_request", None, [], "", "diff_unavailable"),
        ("pull_request", ["unknown/runtime.cc"], [], "", "unmatched_path"),
        (
            "pull_request",
            ["csrc/aten/backends/mthreads/mudnn_copy.cc"],
            [],
            "",
            "watch_paths",
        ),
        ("pull_request", ["README.md"], ["platform:nvidia"], "", "labels"),
        ("pull_request", ["README.md"], ["platform:unknown"], "", "labels"),
        ("workflow_dispatch", ["README.md"], [], "hygon-boxing", "dispatch"),
        ("push", [], [], "", "full_event"),
        ("schedule", [], [], "", "full_event"),
    ],
)
def test_observation_preserves_the_router_matrix(
    event, changed, labels, dispatch, reason
):
    command = [
        sys.executable,
        str(SCRIPTS / "ci/generate_matrix.py"),
        "--event-name",
        event,
        "--configs-dir",
        str(ROOT / ".github/configs"),
        "--pr-labels",
        ",".join(labels),
        "--dispatch-platforms",
        dispatch,
    ]
    if changed is not None:
        command += ["--changed-files", "\n".join(changed)]
    else:
        command += ["--base-sha", "missing-comparison-commit"]
    expected = json.loads(
        subprocess.check_output(
            command, text=True, stderr=subprocess.PIPE
        ).removeprefix("platforms=")
    )
    plan = PLANNER.build_ci_plan(
        request(), CONTRACT, ROOT, changed, event, labels, dispatch
    )
    assert plan["expected_platforms"] == expected
    assert plan["selection_reasons"][-1]["kind"] == reason
    assert plan["expected_shared_checks"] == [
        "lint",
        "agnostic-checks",
        "codegen-checks",
    ]


def test_ci_documentation_does_not_inherit_external_l2_fallback():
    req = request(requested_level="L2")
    observed = PLANNER.build_ci_plan(req, CONTRACT, ROOT, ["README.md"], "pull_request")
    assert observed["expected_platforms"] == observed["execution_plan"] == []
    assert observed["test_level"] == "L1"
    assert PLANNER.build_plan(req, CONTRACT, ROOT, ["README.md"])["expected_platforms"]


def test_plan_hash_is_deterministic_and_ignores_run_and_task_identity():
    first = PLANNER.build_plan(request(), CONTRACT, ROOT, ["setup.py"])
    second = PLANNER.build_plan(
        request(
            run_id="99", run_attempt="3", task_id="another-task", parent_run_id="88"
        ),
        CONTRACT,
        ROOT,
        ["setup.py"],
    )
    assert first["plan_hash"] == second["plan_hash"]
    first["execution_plan"][0]["environment"] = {"A": "1", "B": "2"}
    second["execution_plan"][0]["environment"] = {"B": "2", "A": "1"}
    assert PLANNER.plan_hash(second) == PLANNER.plan_hash(first)


@pytest.mark.parametrize(
    "field",
    ["source_sha", "control_sha", "base_sha", "diff_head_sha", "source_repository"],
)
def test_source_and_control_identity_change_the_hash(field):
    first = PLANNER.build_plan(request(), CONTRACT, ROOT, ["setup.py"])
    second = PLANNER.build_plan(
        request(**{field: "d" * 40}), CONTRACT, ROOT, ["setup.py"]
    )
    assert first["plan_hash"] != second["plan_hash"]


@pytest.mark.parametrize(
    "field", ["command", "environment", "min_passed", "order", "shared_environment"]
)
def test_final_execution_content_is_part_of_the_hash(field):
    plan = PLANNER.build_plan(
        request(platforms="nvidia", requested_level="L2"), CONTRACT, ROOT, []
    )
    altered = copy.deepcopy(plan)
    lane = altered["execution_plan"][0]
    if field == "order":
        lane["groups"].reverse()
    elif field == "shared_environment":
        lane["environment"]["OBSERVED_VALUE"] = "changed"
    elif field == "min_passed":
        lane["groups"][0][field] = 100
    else:
        lane["groups"][0][field] = (
            {"OBSERVED_VALUE": "changed"} if field == "environment" else "changed"
        )
    assert PLANNER.plan_hash(altered) != plan["plan_hash"]


def test_large_execution_plan_stays_out_of_the_dispatch_context():
    plan = PLANNER.build_plan(
        request(run_id="42", run_attempt="1"), CONTRACT, ROOT, None
    )
    context = json.loads(PLANNER.report_context(plan))
    assert "execution_plan" not in context
    assert "selection_reasons" not in context
    assert context["plan_hash"] == plan["plan_hash"]


@pytest.mark.parametrize("selected", [[], ["nvidia"]])
def test_real_observer_cli_records_checkout_identity_and_detects_matrix_drift(
    tmp_path, selected
):
    sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    output = tmp_path / "plan.json"
    env = {
        **os.environ,
        "SOURCE_REPOSITORY": "owner/fork",
        "EVENT_NAME": "pull_request",
        "BASE_SHA": sha,
        "HEAD_SHA": sha,
        "PR_LABELS_JSON": "[]",
        "SELECTED_PLATFORMS": json.dumps(selected),
        "TEST_PROFILE": "pr",
        "DISPATCH_PLATFORMS": "",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "flagcicd_plan.py"),
            "--ci",
            "--source",
            str(ROOT),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if selected:
        assert result.returncode != 0
        assert "differs from the scheduled CI matrix" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        plan = json.loads(output.read_text())
        assert plan["source_sha"] == plan["control_sha"] == sha
        assert plan["observation_only"] is True
        assert plan["execution_plan"] == []


def test_observer_is_nonblocking_and_does_not_replace_the_matrix_or_gate():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    observer = jobs["observe-plan"]
    steps = observer["steps"]
    observe = next(
        s for s in steps if s.get("name") == "Record CI plan (observation only)"
    )
    assert observer["continue-on-error"] is True
    assert "#" not in observe["run"]
    assert (
        observe["env"]["SELECTED_PLATFORMS"]
        == "${{ needs.discover-platforms.outputs.platforms }}"
    )
    assert (
        jobs["discover-platforms"]["outputs"]["platforms"]
        == "${{ steps.detect.outputs.platforms }}"
    )
    assert "flagcicd_plan" not in jobs["platform-pipeline"]["with"]
    assert "flagcicd_report.py" not in json.dumps(jobs["ci-complete"])
    assert "observe-plan" not in jobs["platform-pipeline"]["needs"]
    assert "observe-plan" not in jobs["ci-complete"]["needs"]
    assert not any(
        "flagcicd_plan.py" in s.get("run", "")
        for s in jobs["discover-platforms"]["steps"]
    )


def test_observer_tests_are_conditional_for_older_source_checkouts():
    steps = yaml.safe_load(
        (ROOT / ".github/workflows/agnostic-checks.yml").read_text()
    )["jobs"]["agnostic-checks"]["steps"]
    calls = [
        s for s in steps if "tests/unit/test_ci_plan_observation.py" in s.get("run", "")
    ]
    assert len(calls) == 1
    assert calls[0]["if"] == "hashFiles('tests/unit/test_ci_plan_observation.py') != ''"


def test_observer_diffs_the_pr_head_but_records_the_tested_merge_commit(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(ROOT / ".github/configs", source / ".github/configs")
    shutil.copyfile(
        ROOT / ".github/integration-manifest.yml",
        source / ".github/integration-manifest.yml",
    )

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                *args,
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-qb", "pr")
    (source / "README.md").write_text("documentation")
    git("add", ".")
    git("commit", "-qm", "documentation PR")
    head = git("rev-parse", "HEAD")
    git("checkout", "-q", "main")
    (source / "runtime.cc").write_text("upstream runtime")
    git("add", ".")
    git("commit", "-qm", "upstream runtime")
    git("merge", "--no-ff", "-qm", "merge PR", "pr")
    merged = git("rev-parse", "HEAD")
    output = tmp_path / "plan.json"
    env = {
        **os.environ,
        "SOURCE_REPOSITORY": "owner/base",
        "EVENT_NAME": "pull_request",
        "BASE_SHA": base,
        "HEAD_SHA": head,
        "PR_LABELS_JSON": "[]",
        "SELECTED_PLATFORMS": "[]",
        "TEST_PROFILE": "pr",
        "DISPATCH_PLATFORMS": "",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "flagcicd_plan.py"),
            "--ci",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(output.read_text())
    assert plan["source_sha"] == merged != head
    assert plan["diff_head_sha"] == head
    assert plan["changed_files"] == ["README.md"]
    assert plan["execution_plan"] == []


def test_ci_nightly_is_full_and_weekly_stays_reserved():
    plan = PLANNER.build_ci_plan(
        request(profile="nightly"), CONTRACT, ROOT, [], "schedule"
    )
    configs = ROOT / ".github/configs"
    known = PLANNER._load_platforms(configs)
    assert len(plan["execution_plan"]) == len(known) + len(
        PLANNER.boxing_platforms(configs, known)
    )
    with pytest.raises(ValueError, match="weekly is reserved"):
        PLANNER.build_ci_plan(request(profile="weekly"), CONTRACT, ROOT, [], "schedule")
    with pytest.raises(ValueError, match="nightly requires platforms=all"):
        PLANNER.build_ci_plan(
            request(profile="nightly"),
            CONTRACT,
            ROOT,
            [],
            "workflow_dispatch",
            dispatch="nvidia",
        )
