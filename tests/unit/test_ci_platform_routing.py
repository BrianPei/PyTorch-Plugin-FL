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

"""Keep matrix routing, checkout source and the required gate in agreement.

The gate is no longer an inline shell script: `ci-complete` relays the verdict
of `flagcicd_report.py summary`. These tests execute that step as written, in
a directory laid out like the job (a plan artifact and downloaded platform
reports), so the decision is tested through the same code path a real run
uses -- including the JOB_MAP that translates this workflow's job ids into the
canonical stage names the report checks.
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
WORKFLOWS = ROOT / ".github" / "workflows"
CONFIGS = ROOT / ".github" / "configs"
PLATFORMS = sorted(
    path.stem
    for path in CONFIGS.glob("*.yml")
    if path.stem not in {"template", "image_sources", "platforms"}
)
RUN_ID = "1700000000"


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _gate_step():
    gate = _workflow("ci.yml")["jobs"]["ci-complete"]
    steps = [step for step in gate["steps"] if "flagcicd_report.py" in step.get("run", "")]
    assert len(steps) == 1
    return gate, steps[0]


def test_required_gate_covers_the_plan_and_matrix_pipeline():
    jobs = _workflow("ci.yml")["jobs"]
    gate = jobs["ci-complete"]
    assert gate["name"] == "CI complete"
    assert set(gate["needs"]) == {
        "resolve-checkout",
        "plan",
        "lint",
        "agnostic-checks",
        "codegen-checks",
        "wheel-checks",
        "platform-pipeline",
    }
    # The matrix is the plan's own answer; there is no second discovery job
    # that could disagree with it.
    assert "discover-platforms" not in jobs
    assert (
        jobs["platform-pipeline"]["uses"] == "./.github/workflows/integration-tests.yml"
    )
    assert jobs["platform-pipeline"]["strategy"]["matrix"]["include"] == (
        "${{ fromJson(needs.plan.outputs.matrix) }}"
    )
    assert not list(WORKFLOWS.glob("integration-test-*.yml"))
    assert not (WORKFLOWS / "all-tests-common.yml").exists()


def test_the_matrix_carries_each_platforms_group_selection():
    jobs = _workflow("ci.yml")["jobs"]
    pipeline = jobs["platform-pipeline"]
    assert pipeline["with"]["flagcicd_groups"] == "${{ matrix.groups }}"
    assert pipeline["with"]["flagcicd_plan"] == "${{ needs.plan.outputs.context }}"

    plan_run = next(
        step
        for step in jobs["plan"]["steps"]
        if "flagcicd_plan.py" in step.get("run", "")
    )["run"]
    # The plan job writes `matrix`; the workflow does not recompute it.
    assert "flagcicd_plan.py" in plan_run
    assert "json.loads(open(\"flagcicd-plan.json\")" not in plan_run


def test_platform_tests_and_manifest_use_the_resolved_source():
    caller = _workflow("ci.yml")["jobs"]["platform-pipeline"]
    assert "resolve-checkout" in caller["needs"]
    assert caller["with"]["checkout_repository"] == (
        "${{ needs.resolve-checkout.outputs.repository }}"
    )
    assert caller["with"]["checkout_ref"] == (
        "${{ needs.resolve-checkout.outputs.ref }}"
    )

    workflow = _workflow("integration-tests.yml")
    inputs = workflow[True]["workflow_call"]["inputs"]
    assert {"checkout_repository", "checkout_ref", "flagcicd_groups"} <= inputs.keys()
    config_call = workflow["jobs"]["configuration"]["with"]
    assert config_call["checkout_repository"] == "${{ inputs.checkout_repository }}"
    assert config_call["checkout_ref"] == "${{ inputs.checkout_ref }}"
    assert config_call["flagcicd_groups"] == "${{ inputs.flagcicd_groups }}"
    checkout = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )["with"]
    assert checkout["repository"] == (
        "${{ inputs.checkout_repository || github.repository }}"
    )
    assert checkout["ref"] == "${{ inputs.checkout_ref || github.ref }}"
    assert checkout["persist-credentials"] is False

    config_workflow = _workflow("load-platform-config.yml")
    config_inputs = config_workflow[True]["workflow_call"]["inputs"]
    assert {"checkout_repository", "checkout_ref", "flagcicd_groups"} <= (
        config_inputs.keys()
    )
    config_checkout = next(
        step
        for step in config_workflow["jobs"]["configuration"]["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )["with"]
    assert config_checkout["repository"] == (
        "${{ inputs.checkout_repository || github.repository }}"
    )
    assert config_checkout["ref"] == "${{ inputs.checkout_ref || github.ref }}"


def test_job_map_covers_all_canonical_stages():
    """Verify ci.yml JOB_MAP maps every CANONICAL_JOBS stage."""
    import importlib.util
    import sys

    scripts = ROOT / ".github/scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "flagcicd_report", scripts / "flagcicd_report.py"
        )
        reporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reporter)

        _, step = _gate_step()
        job_map = json.loads(step["env"]["JOB_MAP"])
        canonical = set(reporter.CANONICAL_JOBS)
        mapped = set(job_map.keys())
        assert mapped == canonical, (
            f"JOB_MAP missing: {canonical - mapped}, unexpected: {mapped - canonical}"
        )
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_dispatch_can_select_each_platform(platform):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "ci" / "generate_matrix.py"),
            "--event-name",
            "workflow_dispatch",
            "--configs-dir",
            str(CONFIGS),
            "--dispatch-platforms",
            platform,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout.removeprefix("platforms=")) == [platform]


def _plan(level="L2", platforms=("musa",)):
    return {
        "schema_version": "flagcicd.plan/v1",
        "source_repository": "flagos-ai/Torch-FL",
        "source_sha": "a" * 40,
        "base_sha": "b" * 40,
        "control_sha": "c" * 40,
        "run_id": RUN_ID,
        "run_attempt": "1",
        "profile": "pr",
        "selection": "affected",
        "test_level": level,
        "scope": "installed_wheel_integration",
        "risk_level": "R1",
        "config_hash": "e" * 64,
        "case_set_hash": "f" * 64,
        "task_id": "",
        "parent_run_id": "",
        "platforms": list(platforms),
        "expected_platforms": list(platforms),
        "groups": {platform: [{"id": "unit"}] for platform in platforms},
        "not_selected": {platform: [] for platform in platforms},
    }


def _report(plan, platform, status="passed", attempt=1):
    return {
        "schema_version": "flagcicd.report/v1",
        "report_type": "platform",
        **{
            key: plan[key]
            for key in ("source_repository", "source_sha", "run_id", "test_level", "config_hash")
        },
        "run_attempt": attempt,
        "platform": {"name": platform},
        "status": status,
        "metrics": {},
        "errors": [],
    }


def _run_gate(tmp_path, plan, reports, results, job_map=None):
    scripts = tmp_path / ".github" / "scripts"
    shutil.copytree(ROOT / ".github/scripts", scripts)
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan/flagcicd-plan.json").write_text(json.dumps(plan))
    (tmp_path / "platform-reports").mkdir()
    for report in reports:
        path = tmp_path / "platform-reports" / f"flagcicd-platform-{report['platform']['name']}.json"
        path.write_text(json.dumps(report))
    _, step = _gate_step()
    env = {
        **os.environ,
        "JOB_RESULTS": json.dumps(results),
        "JOB_MAP": json.dumps(job_map) if job_map is not None else step["env"]["JOB_MAP"],
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": RUN_ID,
    }
    process = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return process, (tmp_path / "flagcicd-run-summary.json")


def _results(level="L2", **overrides):
    results = {
        name: {"result": "success"}
        for name in (
            "resolve-checkout",
            "plan",
            "lint",
            "agnostic-checks",
            "codegen-checks",
            "wheel-checks",
            "platform-pipeline",
        )
    }
    if level == "L0":
        for name in ("agnostic-checks", "codegen-checks", "wheel-checks", "platform-pipeline"):
            results[name]["result"] = "skipped"
    results.update(overrides)
    return results


def test_the_gate_passes_when_every_planned_outcome_is_present(tmp_path):
    plan = _plan()
    process, summary = _run_gate(
        tmp_path, plan, [_report(plan, "musa")], _results()
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(summary.read_text())["status"] == "passed"


def test_a_failed_platform_pipeline_fails_the_gate(tmp_path):
    plan = _plan()
    process, summary = _run_gate(
        tmp_path,
        plan,
        [_report(plan, "musa")],
        _results(**{"platform-pipeline": {"result": "failure"}}),
    )
    assert process.returncode == 1
    payload = json.loads(summary.read_text())
    assert payload["status"] != "passed"
    assert any("platforms: expected success" in error for error in payload["errors"])


def test_a_platform_that_never_reported_fails_the_gate(tmp_path):
    plan = _plan()
    process, summary = _run_gate(tmp_path, plan, [], _results())
    assert process.returncode == 1
    assert any(
        "missing platform report: musa" in error
        for error in json.loads(summary.read_text())["errors"]
    )


def test_an_l0_plan_does_not_owe_hardware_or_agnostic_work(tmp_path):
    plan = _plan(level="L0", platforms=())
    process, summary = _run_gate(tmp_path, plan, [], _results(level="L0"))
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(summary.read_text())["status"] == "passed"


def test_a_job_map_that_names_a_renamed_job_fails_by_name(tmp_path):
    plan = _plan()
    job_map = {
        "plan": "plan",
        "preflight": "plan",
        "lint": "lint",
        "agnostic": "agnostic-checks",
        "codegen": "codegen-checks",
        "platforms": "platform-pipeline-v2",
    }
    process, summary = _run_gate(
        tmp_path, plan, [_report(plan, "musa")], _results(), job_map
    )
    assert process.returncode == 1
    assert any(
        "platform-pipeline-v2" in error
        for error in json.loads(summary.read_text())["errors"]
    )


@pytest.mark.parametrize(
    "name",
    ["plan", "lint", "agnostic-checks", "codegen-checks", "wheel-checks", "platform-pipeline"],
)
def test_required_gate_rejects_skipped_shared_checks(tmp_path, name):
    plan = _plan()
    process, summary = _run_gate(
        tmp_path,
        plan,
        [_report(plan, "musa")],
        _results(**{name: {"result": "skipped"}}),
    )
    assert process.returncode == 1
    errors = json.loads(summary.read_text())["errors"]
    stage = {
        "agnostic-checks": "agnostic",
        "codegen-checks": "codegen",
        "wheel-checks": "wheel",
        "platform-pipeline": "platforms",
    }.get(name, name)
    assert any(f"{stage}: expected success, received skipped" == error for error in errors)


def test_the_gate_still_reports_when_the_plan_job_failed(tmp_path):
    """A missing plan must be a failed run, not an empty one that passes."""
    scripts = tmp_path / ".github" / "scripts"
    shutil.copytree(ROOT / ".github/scripts", scripts)
    (tmp_path / "plan").mkdir()
    (tmp_path / "platform-reports").mkdir()
    (tmp_path / "event.json").write_text(json.dumps({"inputs": {"task_id": "t-1"}}))
    _, step = _gate_step()
    env = {
        **os.environ,
        "JOB_RESULTS": json.dumps(_results(**{"plan": {"result": "failure"}})),
        "JOB_MAP": json.dumps(json.loads(step["env"]["JOB_MAP"])),
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": RUN_ID,
        "GITHUB_REPOSITORY": "flagos-ai/Torch-FL",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json"),
    }
    process = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    payload = json.loads((tmp_path / "flagcicd-run-summary.json").read_text())
    assert payload["status"] == "infra_error"
    assert payload["task_id"] == "t-1"
