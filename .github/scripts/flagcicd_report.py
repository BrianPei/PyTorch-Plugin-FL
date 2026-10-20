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

"""Describe platform execution and aggregate a plan without combining coverage."""

import argparse
import json
import math
import os
from pathlib import Path

from flagcicd_resolve import text as request_text


SCHEMA = "flagcicd.report/v1"
FAILURES = {"build_failed", "test_failed", "infra_error", "cancelled", "report_error"}
IDENTITY = ("source_repository", "source_sha", "run_id", "test_level", "config_hash")
REQUIRED_STAGES = ("setup", "build", "validate", "install", "tests")

# The stages a run is judged by, independent of how the workflow that ran it
# names its jobs. `preflight` and `plan` are separate stages because a plan
# that never resolves and a plan that resolves to nothing are different
# failures, even when one job produces both.
CANONICAL_JOBS = ("plan", "preflight", "lint", "agnostic", "codegen", "wheel", "platforms")


def identity(plan):
    return {
        "repository": plan["source_repository"],
        **{
            key: plan[key]
            for key in (*IDENTITY, "scope", "risk_level", "task_id", "parent_run_id")
        },
        "level_deferred": plan.get("level_deferred"),
    }


def failed_request_identity():
    """Retain safe correlation IDs when no validated source plan is available."""
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            request = event.get("inputs")
        elif (
            os.environ.get("GITHUB_EVENT_NAME") == "repository_dispatch"
            and event.get("action") == "flagcicd"
        ):
            request = event.get("client_payload")
        else:
            return {}
    except (KeyError, OSError, ValueError, AttributeError):
        return {}
    if not isinstance(request, dict):
        return {}
    fields = {}
    for key in ("task_id", "parent_run_id"):
        try:
            fields[key] = request_text(request.get(key, ""), key)
        except ValueError:
            continue
    return fields


def one_report(directory, filename):
    paths = list(directory.rglob(filename))
    if len(paths) != 1:
        raise ValueError(f"expected one {filename}, found {len(paths)}")
    payload = json.loads(paths[0].read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{filename} must contain an object")
    return payload


GROUP_STATUSES = {"passed", "failed", "blocked", "timed_out"}
GROUPS_SCHEMA = "flagcicd.groups/v1"


def _group_paths(paths):
    """Index the runner's group records by id, refusing anything ambiguous."""
    records = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != GROUPS_SCHEMA
        ):
            raise ValueError(f"{path.name}: unsupported group record schema")
        groups = payload.get("groups")
        if not isinstance(groups, list):
            raise ValueError(f"{path.name}: groups must be a list")
        for record in groups:
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}: a group record must be an object")
            group_id = record.get("id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{path.name}: a group record has no id")
            if record.get("status") not in GROUP_STATUSES:
                raise ValueError(f"{group_id}: invalid group status")
            if group_id in records:
                raise ValueError(f"{group_id}: group reported more than once")
            records[group_id] = record
    return records


def _check_groups(records, selected, platform):
    """Every selected group ran, and nothing else did.

    A selection is a promise about what this run proves. Completion needs both
    directions: a group the plan selected but the runner never reported means
    the gate would pass on evidence that does not exist, and a group the
    runner reported but the plan did not select means the runner ran work the
    plan does not account for. Both are report errors, not skips.

    Blocked and timed-out are reported as their own statuses rather than
    folded into "failed": a blocked group was never attempted, and a timeout
    is a budget decision, and a reader deciding whether to retest needs to
    tell them apart. For the gate, neither passes.
    """
    errors = []
    expected = {group["id"] for group in selected}
    missing = sorted(expected - set(records))
    unexpected = sorted(set(records) - expected)
    if missing:
        errors.append(
            f"selected groups never reported on {platform}: {', '.join(missing)}"
        )
    if unexpected:
        errors.append(
            f"groups reported but not selected on {platform}: {', '.join(unexpected)}"
        )
    for group_id in sorted(expected & set(records)):
        status = records[group_id]["status"]
        if status == "passed":
            continue
        detail = records[group_id].get("detail") or ""
        errors.append(f"{group_id}: {status}{f' ({detail})' if detail else ''}")
    return errors


def platform_report(plan, platform, jobs, raw, attempt, groups_path=None):
    if platform not in plan["expected_platforms"]:
        raise ValueError("platform is not expected by the plan")
    job = jobs["integration-tests"]
    stages = json.loads(job.get("outputs", {}).get("stages") or "{}")
    result = job["result"]
    errors = []

    # ci-test-tiers.md line 13: L2+ runs must record hardware identity
    hardware_identity = {}
    if plan["test_level"] not in ("L0", "L1"):
        hardware_path = raw / "pytest" / f"hardware-{platform}.json"
        try:
            hardware_identity = json.loads(hardware_path.read_text())
            if not all(k in hardware_identity for k in ("machine_model", "sdk_version", "accelerator_count")):
                errors.append(f"hardware identity file missing required fields: {hardware_path}")
        except OSError:
            errors.append(f"L2+ run missing hardware identity file: {hardware_path}")

    if jobs["configuration"]["result"] == "cancelled" or result == "cancelled":
        status = "cancelled"
    elif jobs["configuration"]["result"] != "success":
        status = "infra_error"
    elif any(stages.get(key) == "failure" for key in ("build", "validate", "install")):
        status = "build_failed"
    elif stages.get("setup") == "failure":
        status = "infra_error"
    elif stages.get("tests") == "failure":
        status = "test_failed"
    elif result == "success" and all(
        stages.get(key) == "success" for key in REQUIRED_STAGES
    ):
        status = "passed"
    else:
        status = "infra_error"
        errors.append("the platform did not successfully complete all required stages")
    metrics = {"coverage": {"available": False}}
    try:
        pytest = one_report(raw / "pytest", f"pytest-{platform}.json")
        metadata = pytest["flagcicd"]
        if (
            metadata["source_repository"] != plan["source_repository"]
            or metadata["source_ref"] != plan["source_sha"]
            or metadata["platform"] != platform
        ):
            raise ValueError(
                "pytest report does not match the planned source and platform"
            )
        if metadata["complete"] is not True:
            raise ValueError("pytest report is incomplete")
        counts = pytest["summary"]
        if not isinstance(counts, dict) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts.values()
        ):
            raise ValueError("invalid pytest summary")
        if status == "passed" and pytest["exitcode"] != 0:
            raise ValueError("pytest report contradicts the successful job result")
        if status == "passed" and any(
            counts.get(key, 0) for key in ("failed", "error")
        ):
            raise ValueError(
                "pytest failure counts contradict the successful job result"
            )
        metrics["tests"] = counts
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
        if status == "passed":
            status = "report_error"

    selected = plan["groups"][platform]
    group_metrics = {"expected": len(selected), "reported": 0, "statuses": {}}
    try:
        if groups_path is None:
            raise ValueError("the runner did not retain group records")
        records = _group_paths([groups_path])
        group_metrics["reported"] = len(records)
        group_metrics["statuses"] = {
            group_id: record["status"] for group_id, record in sorted(records.items())
        }
        group_errors = _check_groups(records, selected, platform)
        if status == "passed" and group_errors:
            raise ValueError(
                "group records contradict the successful job result: "
                + "; ".join(group_errors)
            )
        errors.extend(group_errors)
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
        if status == "passed":
            status = "report_error"
    metrics["groups"] = group_metrics
    coverage_errors = []
    try:
        coverage = one_report(raw / "coverage", f"coverage-{platform}.json")
        metadata = one_report(raw / "coverage", "metadata.json")
        if (
            metadata["source_repository"] != plan["source_repository"]
            or metadata["source_ref"] != plan["source_sha"]
            or metadata["platform"] != platform
            or metadata["complete"] is not True
        ):
            raise ValueError("coverage is incomplete or belongs to another source")
        totals = coverage["totals"]
        covered, total = totals["covered_lines"], totals["num_statements"]
        if (
            not all(type(value) is int for value in (covered, total))
            or not 0 <= covered <= total
        ):
            raise ValueError("invalid coverage line counts")
        percent = totals["percent_covered"]
        if (
            not isinstance(percent, (int, float))
            or not math.isfinite(percent)
            or not 0 <= percent <= 100
        ):
            raise ValueError("invalid coverage percentage")
        metrics["coverage"] = {
            "available": True,
            "scope": "installed torch_fl Python files",
            "metric_type": "line",
            "covered": covered,
            "total": total,
            "percent": percent,
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        coverage_errors.append(str(error))
    return {
        "schema_version": SCHEMA,
        "report_type": "platform",
        **identity(plan),
        "run_attempt": attempt,
        "platform": {"name": platform, **hardware_identity},
        "status": status,
        "configuration_result": jobs["configuration"]["result"],
        "execution_result": result,
        "execution_attempt": job.get("outputs", {}).get("run_attempt", ""),
        "stages": stages,
        "metrics": metrics,
        "errors": errors,
        "coverage_warnings": coverage_errors,
    }


def parse_job_map(raw):
    """Read the stage -> job id map a workflow publishes as JOB_MAP.

    Absent, the identities hold: a workflow whose job ids already are the
    canonical stage names needs no map. Present, it must be a flat object of
    non-empty strings; a typo there is a configuration error the gate should
    report, not a key it should tolerate.
    """
    if not raw:
        return {name: name for name in CANONICAL_JOBS}
    try:
        mapping = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"JOB_MAP is not JSON: {error}") from error
    if not isinstance(mapping, dict):
        raise ValueError("JOB_MAP must be an object")
    for stage, job_id in mapping.items():
        if not isinstance(stage, str) or not isinstance(job_id, str) or not job_id:
            raise ValueError("JOB_MAP must map stage names to job ids")
    return mapping


def job_results(jobs, raw):
    """Resolve every canonical stage to the result of the job that ran it.

    The report names stages the way the plan does; how a workflow names its
    jobs is local to that workflow. This map is the only place the two
    vocabularies meet. A job renamed without updating the map therefore fails
    here by name, instead of matching nothing and letting the check the stage
    stands for pass on a result that was never produced.
    """
    mapping = parse_job_map(raw)
    errors = []
    unknown = sorted(set(mapping) - set(CANONICAL_JOBS))
    if unknown:
        errors.append(f"JOB_MAP names unknown stages: {', '.join(unknown)}")
    results = {}
    for name in CANONICAL_JOBS:
        job_id = mapping.get(name)
        if job_id is None:
            errors.append(f"JOB_MAP does not map the {name} stage")
            continue
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            errors.append(
                f"{name}: JOB_MAP points at job {job_id!r}, which this run does not have"
            )
            continue
        results[name] = job.get("result", "missing")
    return results, errors


def aggregate(plan, jobs, directory, attempt, job_map="", plan_error=""):
    errors, latest, seen, duplicates, ignored = [], {}, set(), set(), []
    if plan is None:
        errors.append(plan_error or "test plan is unavailable")
        plan = {"expected_platforms": []}
    level = plan.get("test_level", "L0")
    expected_jobs = {"plan", "lint", "preflight"}
    if level != "L0":
        expected_jobs.update(("agnostic", "codegen", "wheel"))
    if level in ("L2", "L3", "L4"):
        expected_jobs.add("platforms")
        if not plan.get("expected_platforms"):
            errors.append(f"{level} plans no platform, so it proves nothing")
    try:
        stages, map_errors = job_results(jobs, job_map)
    except ValueError as error:
        stages, map_errors = {}, [str(error)]
    errors.extend(map_errors)
    for name in CANONICAL_JOBS:
        if name not in stages:
            continue
        expected = "success" if name in expected_jobs else "skipped"
        if stages[name] != expected:
            errors.append(f"{name}: expected {expected}, received {stages[name]}")
    for path in sorted(directory.rglob("flagcicd-platform-*.json")):
        try:
            report = json.loads(path.read_text())
            if (
                report["schema_version"] != SCHEMA
                or report["report_type"] != "platform"
            ):
                raise ValueError("unsupported platform report schema")
            report_attempt = report["run_attempt"]
            if type(report_attempt) is not int or not 1 <= report_attempt <= attempt:
                raise ValueError("invalid report attempt")
            matches = all(report[key] == plan[key] for key in IDENTITY)
            if not matches:
                if report_attempt < attempt and report["run_id"] == plan["run_id"]:
                    ignored.append(
                        {
                            "path": path.name,
                            "reason": "previous attempt used a different plan",
                        }
                    )
                    continue
                raise ValueError("report identity does not match the plan")
            platform = report["platform"]["name"]
            if platform not in plan["expected_platforms"]:
                raise ValueError("report platform is not expected")
            if report["status"] not in FAILURES | {"passed"}:
                raise ValueError("invalid report status")
            key = (platform, report_attempt)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
            previous = latest.get(platform)
            if previous is None or previous["run_attempt"] < report_attempt:
                latest[platform] = report
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{path.name}: {error}")
    for platform in plan["expected_platforms"]:
        if platform not in latest:
            errors.append(f"missing platform report: {platform}")
        elif (platform, latest[platform]["run_attempt"]) in duplicates:
            errors.append(f"duplicate latest platform report: {platform}")
    statuses = [report["status"] for report in latest.values()]
    status = "passed"
    for candidate in (
        "cancelled",
        "test_failed",
        "build_failed",
        "infra_error",
        "report_error",
    ):
        if candidate in statuses:
            status = candidate
    for name in CANONICAL_JOBS:
        if name not in expected_jobs or name not in stages:
            continue
        result = stages[name]
        if result == "failure":
            if name in ("plan", "preflight"):
                status = "infra_error"
            elif name != "platforms":
                status = "test_failed"
            elif status == "passed":
                # All expected reports are present and passing, yet the matrix
                # job failed: the failure sits outside the stages the report
                # covers (a child workflow's own failure, a setup step before
                # the stages, an unusable runner). That is infra, not a report
                # defect; a missing or contradictory report is caught above.
                status = "infra_error"
        elif result == "cancelled" and status == "passed":
            status = "cancelled"
        elif result in ("missing", "skipped") and status == "passed":
            status = "infra_error"
    if errors and status == "passed":
        status = "report_error" if stages.get("plan") == "success" else "infra_error"
    return {
        "schema_version": SCHEMA,
        "report_type": "run_summary",
        "platform": {"name": "multi"},
        "repository": plan.get("source_repository", ""),
        **{
            key: plan.get(key, "")
            for key in (*IDENTITY, "scope", "risk_level", "task_id", "parent_run_id")
        },
        "level_deferred": plan.get("level_deferred"),
        "run_attempt": attempt,
        "status": status,
        "expected_platforms": plan["expected_platforms"],
        "jobs": {name: job["result"] for name, job in jobs.items()},
        "stages": stages,
        "children": [latest[key] for key in sorted(latest)],
        "coverage_aggregation": "per_platform_only",
        "errors": errors,
        "ignored_reports": ignored,
    }


def load_plan(path):
    """The full plan, from the retained artifact when one was downloaded.

    The artifact is the only copy that carries ``groups``: the environment
    context is deliberately truncated to the fields a report needs, because a
    plan with a large diff does not fit in a dispatch payload. Summary mode
    falls back to that context, which is enough to judge stages and expected
    platforms but not enough to judge groups -- and a call that wants group
    evidence without a plan artifact says so rather than passing silently.
    """
    if path is None:
        raw = os.environ.get("FLAGCICD_PLAN", "")
        return json.loads(raw) if raw else None
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{Path(path).name} must contain an object")
    if payload.get("schema_version") != "flagcicd.plan/v1":
        raise ValueError(f"{Path(path).name}: unsupported plan schema")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("platform", "summary"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="")
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help="retained flagcicd-plan.json, which carries groups; "
        "without it the compact FLAGCICD_PLAN context is used",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="the runner's group records flagcicd.groups/v1",
    )
    args = parser.parse_args()
    plan, plan_error = None, ""
    try:
        plan = load_plan(args.plan_file)
    except (OSError, ValueError, TypeError) as error:
        plan_error = f"the plan is unavailable: {error}"
    jobs = json.loads(os.environ["JOB_RESULTS"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    if args.mode == "platform":
        if plan is None:
            print(f"::error::{plan_error or 'test plan is unavailable'}")
            return 2
        if plan_error:
            print(f"::warning::{plan_error}")
        report = platform_report(
            plan, args.platform, jobs, args.directory, attempt, args.groups
        )
    else:
        report = aggregate(
            plan,
            jobs,
            args.directory,
            attempt,
            os.environ.get("JOB_MAP", ""),
            plan_error,
        )
        report["run_id"] = os.environ["GITHUB_RUN_ID"]
        if not plan:
            report["repository"] = os.environ.get("GITHUB_REPOSITORY", "")
            report.update(failed_request_identity())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"FlagCICD {report['report_type']}: {report['status']}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
