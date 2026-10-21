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
CONTEXT = ("profile", "control_sha", "target_kind", "base_sha", "diff_sha", "selection")


def identity(plan):
    return {
        "repository": plan["source_repository"],
        **{
            key: plan[key]
            for key in (*IDENTITY, "scope", "risk_level", "task_id", "parent_run_id")
        },
        **{key: plan.get(key, "") for key in CONTEXT},
    }


def entry_jobs(jobs, entry):
    """Map entry-specific job IDs onto the shared verdict contract."""
    if entry == "external":
        return jobs
    aliases = {
        "plan": "discover-platforms",
        "preflight": "preflight",
        "lint": "lint",
        "agnostic": "agnostic-checks",
        "codegen": "codegen-checks",
        "wheel-checks": "wheel-checks",
        "platforms": "platform-pipeline",
    }
    mapped = {
        name: jobs.get(alias, {"result": "missing"}) for name, alias in aliases.items()
    }
    resolution = jobs.get("resolve-checkout", {"result": "missing"})
    if resolution["result"] != "success":
        mapped["plan"] = resolution
    return mapped


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


def validate_case_contracts(plan, execution, hardware):
    """Independently reject successful evidence that contradicts its plan."""
    contracts = plan.get("case_contracts")
    if contracts is None:
        return
    if not isinstance(contracts, dict):
        raise ValueError("invalid planned case contracts")
    for group in execution.get("groups", []):
        if not isinstance(group, dict):
            raise ValueError("invalid executed case evidence")
        case_id = group.get("id")
        contract = contracts.get(case_id)
        if not isinstance(contract, dict):
            raise ValueError(f"missing planned execution contract: {case_id}")
        minimum = contract.get("min_accelerators")
        required_passes = contract.get("min_passed")
        if (
            type(minimum) is not int
            or minimum < 1
            or type(required_passes) is not int
            or required_passes < 0
        ):
            raise ValueError(f"invalid execution contract: {case_id}")
        resources = group.get("resources")
        if (
            not isinstance(resources, dict)
            or type(resources.get("min_accelerators")) is not int
            or resources.get("min_accelerators") != minimum
        ):
            raise ValueError(f"executed resources do not match the plan: {case_id}")
        count = hardware.get("accelerator_count")
        if type(count) is not int or count < minimum:
            raise ValueError(f"insufficient recorded resources for case: {case_id}")
        if group.get("status") != "passed":
            raise ValueError(
                f"non-passing case contradicts successful execution: {case_id}"
            )
        if required_passes:
            counts = group.get("summary")
            if not isinstance(counts, dict) or not all(
                type(value) is int and value >= 0 for value in counts.values()
            ):
                raise ValueError(f"missing or invalid case counts: {case_id}")
            if counts.get("passed", 0) < required_passes or any(
                counts.get(key, 0) for key in ("failed", "error")
            ):
                raise ValueError(
                    f"required case has no clean passing evidence: {case_id}"
                )


def platform_report(plan, platform, jobs, raw, attempt):
    if platform not in plan["expected_platforms"]:
        raise ValueError("platform is not expected by the plan")
    job = jobs["integration-tests"]
    result = job["result"]
    errors = []
    try:
        stages = json.loads(job.get("outputs", {}).get("stages") or "{}")
        if not isinstance(stages, dict):
            raise ValueError("stage evidence must be an object")
    except (ValueError, TypeError) as error:
        stages = {}
        errors.append(f"invalid stage evidence: {error}")

    # Record hardware identity for L2+ runs
    hardware_identity = {}
    if plan["test_level"] not in ("L0", "L1"):
        try:
            hardware_identity = one_report(raw / "pytest", f"hardware-{platform}.json")
            if not all(
                k in hardware_identity
                for k in ("machine_model", "sdk_version", "accelerator_count")
            ):
                raise ValueError("hardware identity is missing required fields")
            if (
                type(hardware_identity["accelerator_count"]) is not int
                or hardware_identity["accelerator_count"] < 1
            ):
                raise ValueError("hardware execution requires at least one accelerator")
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"invalid L2+ hardware identity: {error}")

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
    if errors and status == "passed":
        status = "report_error"
    execution = {}
    ledgers = list((raw / "pytest").rglob("groups.json"))
    if ledgers:
        try:
            execution = one_report(raw / "pytest", "groups.json")
            blocked = [
                group
                for group in execution.get("groups", [])
                if group.get("status") == "blocked"
            ]
            if blocked:
                if status not in ("cancelled", "build_failed"):
                    status = "infra_error"
                errors.extend(
                    f"blocked case {group.get('id')}: {group.get('reason')}"
                    for group in blocked
                )
        except (OSError, ValueError, TypeError, AttributeError) as error:
            execution = {}
            errors.append(f"invalid execution ledger: {error}")
            if status == "passed":
                status = "report_error"
    metrics = {"coverage": {"available": False}}
    if list((raw / "pytest").rglob("performance.json")):
        try:
            metrics["performance"] = one_report(raw / "pytest", "performance.json")
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"invalid performance evidence: {error}")
            if status == "passed":
                status = "report_error"
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
        counts = pytest["summary"]
        if not isinstance(counts, dict) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts.values()
        ):
            raise ValueError("invalid pytest summary")
        metrics["tests"] = counts
        provided_execution = metadata.get("execution")
        if provided_execution is not None:
            if not isinstance(provided_execution, dict):
                raise ValueError("execution evidence must be an object")
            groups = provided_execution.get("groups", [])
            if not isinstance(groups, list):
                raise ValueError("execution groups must be an array")
            if not all(isinstance(group, dict) for group in groups):
                raise ValueError("execution groups must contain objects")
            execution = provided_execution
        if metadata["complete"] is not True:
            raise ValueError("pytest report is incomplete")
        expected_cases = plan.get("expected_cases", {}).get(platform)
        if plan.get("case_contracts") is not None and expected_cases is None:
            raise ValueError("planned case sequence is missing")
        if expected_cases is not None:
            execution = metadata.get("execution") or {}
            observed_cases = [group["id"] for group in execution.get("groups", [])]
            if (
                execution.get("complete") is not True
                or observed_cases != expected_cases
            ):
                raise ValueError(
                    "executed case groups do not match the trusted test plan"
                )
        if status == "passed":
            validate_case_contracts(plan, execution, hardware_identity)
        if status == "passed" and pytest["exitcode"] != 0:
            raise ValueError("pytest report contradicts the successful job result")
        if status == "passed" and any(
            counts.get(key, 0) for key in ("failed", "error")
        ):
            raise ValueError(
                "pytest failure counts contradict the successful job result"
            )
        if status == "passed" and counts.get("passed", 0) < 1:
            raise ValueError("hardware execution requires at least one passing test")
        metrics["tests"] = counts
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
        if status == "passed":
            status = "report_error"
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
        "execution": execution,
        "unsupported_cases": [
            case
            for case in plan.get("unsupported_cases", [])
            if case["platform"] == platform
        ],
        "metrics": metrics,
        "errors": errors,
        "coverage_warnings": coverage_errors,
    }


def aggregate(plan, jobs, directory, attempt):
    errors, latest, seen, duplicates, ignored = [], {}, set(), set(), []
    if plan is None:
        errors.append("test plan is unavailable")
        plan = {"expected_platforms": []}
    level = plan.get("test_level", "L0")
    for blocked in plan.get("blocked_cases", []):
        errors.append(
            f"blocked planned case: {blocked['platform']}/{blocked['id']}: {blocked['reason']}"
        )
    expected_jobs = {"plan", "lint", "preflight"}
    if level in ("L1", "L2", "L3", "L4"):
        expected_jobs.update(("agnostic", "codegen", "wheel-checks"))
    if level in ("L2", "L3", "L4"):
        expected_jobs.add("platforms")
    for name in (
        "plan",
        "lint",
        "preflight",
        "agnostic",
        "codegen",
        "wheel-checks",
        "platforms",
    ):
        result = jobs.get(name, {}).get("result", "missing")
        expected = "success" if name in expected_jobs else "skipped"
        if result != expected:
            errors.append(f"{name}: expected {expected}, received {result}")
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
            matches = all(report[key] == plan[key] for key in IDENTITY) and all(
                report.get(key, "") == plan.get(key, "") for key in CONTEXT
            )
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
    for name in (
        "plan",
        "preflight",
        "lint",
        "agnostic",
        "codegen",
        "wheel-checks",
        "platforms",
    ):
        if name not in expected_jobs:
            continue
        result = jobs.get(name, {}).get("result", "missing")
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
        status = (
            "report_error"
            if jobs.get("plan", {}).get("result") == "success"
            else "infra_error"
        )
    return {
        "schema_version": SCHEMA,
        "report_type": "run_summary",
        "platform": {"name": "multi"},
        "repository": plan.get("source_repository", ""),
        **{
            key: plan.get(key, "")
            for key in (*IDENTITY, "scope", "risk_level", "task_id", "parent_run_id")
        },
        **{key: plan.get(key, "") for key in CONTEXT},
        "run_attempt": attempt,
        "status": status,
        "expected_platforms": plan["expected_platforms"],
        "jobs": {name: job["result"] for name, job in jobs.items()},
        "children": [latest[key] for key in sorted(latest)],
        "coverage_aggregation": "per_platform_only",
        "errors": errors,
        "ignored_reports": ignored,
        **{
            key: plan.get(key, default)
            for key, default in (
                ("validation_scope", "full"),
                ("selection_origin", ""),
                ("required_platforms", []),
                ("blocked_cases", []),
                ("unsupported_cases", []),
                ("manual_test_count", 0),
                ("platform_decisions", []),
                ("plan_artifact", ""),
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("platform", "summary"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="")
    parser.add_argument("--entry", choices=("external", "ci"), default="external")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    if args.record_only and args.mode != "platform":
        parser.error(
            "only platform reports may use --record-only; summary always gates"
        )
    raw_plan = os.environ.get("FLAGCICD_PLAN", "")
    plan = json.loads(raw_plan) if raw_plan else None
    jobs = json.loads(os.environ["JOB_RESULTS"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    if args.mode == "platform":
        report = platform_report(plan, args.platform, jobs, args.directory, attempt)
    else:
        report = aggregate(plan, entry_jobs(jobs, args.entry), args.directory, attempt)
        if args.entry == "ci" and report["validation_scope"] == "diagnostic":
            report["errors"].append(
                "diagnostic platform selection cannot satisfy the full PR gate"
            )
            if report["status"] == "passed":
                report["status"] = "report_error"
        report["entry_jobs"] = {name: job["result"] for name, job in jobs.items()}
        report["run_id"] = os.environ["GITHUB_RUN_ID"]
        if not plan:
            report["repository"] = os.environ.get("GITHUB_REPOSITORY", "")
            report.update(failed_request_identity())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"FlagCICD {report['report_type']}: {report['status']}")
    # A failed hardware job already supplies a root failure for bounded retries.
    # Report defects after successful hardware must fail here so rerun-failed can
    # regenerate the evidence rather than repeatedly rerunning only the summary.
    if args.record_only and any(
        job.get("result") in ("failure", "cancelled") for job in jobs.values()
    ):
        return 0
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
