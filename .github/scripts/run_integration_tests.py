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

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

# Failure policy semantics:
#   fail-fast (default)        -- a failure aborts the remaining entries. Use for
#                                 environment preflight entries (device node,
#                                 wheel presence) where continuing is meaningless.
#   continue-after-failure     -- a failure is recorded and the remaining entries
#                                 still run. Use for functional test groups where
#                                 one group failing must not hide the others:
#                                 the summary reports every failed group so the
#                                 cross-platform evidence is preserved.
DEFAULT_FAILURE_POLICY = "fail-fast"

# Directory (under the integration workdir) that receives generated reports.
REPORT_DIR = "integration-reports"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def _entry_id(entry, index):
    entry_id = entry.get("id")
    if isinstance(entry_id, str) and entry_id.strip():
        return entry_id.strip()
    # Deterministic fallback derived from the name, so the summary is stable
    # across runs even when a manifest omits an explicit id.
    name = entry.get("name", f"test-{index}")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or f"test-{index}"


def _entry_phase(entry, index):
    phase = entry.get("phase")
    if isinstance(phase, str) and phase.strip():
        return phase.strip()
    # Environment/preflight entries fail fast; functional groups continue.
    name = entry.get("name", "")
    lowered = name.lower()
    preflight_markers = ("check ", "availability", "isolated", "environment")
    if any(marker in lowered for marker in preflight_markers):
        return "preflight"
    return "functional"


def _entry_failure_policy(entry, index):
    policy = entry.get("failure_policy")
    if isinstance(policy, str) and policy.strip():
        return policy.strip()
    # Preflight entries default to fail-fast; functional entries default to
    # continue-after-failure so one functional group failing does not hide others.
    phase = _entry_phase(entry, index)
    if phase == "functional":
        return "continue-after-failure"
    return "fail-fast"


def load_configuration(allow_empty):
    try:
        tests = json.loads(os.environ.get("INTEGRATION_TESTS", "[]"))
        environment = json.loads(os.environ.get("INTEGRATION_ENVIRONMENT", "{}"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid integration configuration JSON: {error}") from error

    if not isinstance(tests, list):
        raise SystemExit("integration_tests must be an array")
    if not allow_empty and not tests:
        raise SystemExit("integration_tests must not be empty")
    for index, test in enumerate(tests, start=1):
        if not isinstance(test, dict):
            raise SystemExit(f"integration_tests[{index}] must be an object")
        name = test.get("name")
        command = test.get("command")
        if (
            not isinstance(name, str)
            or not name.strip()
            or "\n" in name
            or "\r" in name
        ):
            raise SystemExit(
                f"integration_tests[{index}].name must be a non-empty single-line string"
            )
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise SystemExit(
                f"integration_tests[{index}].command must be a non-empty string"
            )
        for field in ("id", "phase", "failure_policy"):
            value = test.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SystemExit(
                    f"integration_tests[{index}].{field} must be a non-empty string"
                )

    if not isinstance(environment, dict):
        raise SystemExit("integration_environment must be an object")
    for key, value in environment.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f"invalid environment variable name: {key}")
        if (
            not isinstance(value, str)
            or "\n" in value
            or "\r" in value
            or "\0" in value
        ):
            raise SystemExit(
                f"integration_environment[{key}] must be a single-line string"
            )

    # Reject configurations where every entry uses continue-after-failure (no
    # fail-fast gate). At least one entry must abort on failure so preflight
    # failures (missing device, broken wheel) do not run the entire functional
    # suite and produce misleading aggregate results.
    if tests and all(
        _entry_failure_policy(test, idx) == "continue-after-failure"
        for idx, test in enumerate(tests, start=1)
    ):
        raise SystemExit(
            "integration_tests must have at least one fail-fast entry "
            "(preflight checks should abort on failure)"
        )

    return tests, environment


def _command_is_pytest(command):
    # Detect a pytest invocation so per-group JUnit can be emitted without
    # changing the manifest command text. Handles both the `pytest ...` form and
    # the `python -m pytest ...` form.
    words = shlex.split(command)
    for index, word in enumerate(words):
        if word == "pytest" or word.endswith("/pytest"):
            return True
        if word == "-m" and index + 1 < len(words) and words[index + 1] == "pytest":
            return True
    return False


def run_entry(test, command_environment, workdir, report_root, index, total):
    name = test["name"]
    command = test["command"]
    entry_id = _entry_id(test, index)
    phase = _entry_phase(test, index)
    policy = _entry_failure_policy(test, index)

    log_path = report_root / "integration.log"
    junit_path = report_root / f"junit-{entry_id}.xml"

    print(
        f"===== [{index}/{total}] {name} =====",
        flush=True,
    )

    # Prepend the JUnit path for pytest commands so each group produces its own
    # XML, while leaving the manifest command otherwise untouched.
    run_command = command
    if _command_is_pytest(command) and not _has_junitxml(command):
        run_command = f"{command} --junitxml={junit_path}"

    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n===== [{index}/{total}] {name} =====\n")
        log_fh.flush()
        result = subprocess.run(
            ["bash", "-c", f"set -euo pipefail\n{run_command}"],
            check=False,
            env=command_environment,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = result.stdout.decode("utf-8", errors="replace")
        log_fh.write(output)
        if result.returncode:
            log_fh.write(f"\n--- {name} FAILED (exit code {result.returncode}) ---\n")
        log_fh.flush()

    duration = time.monotonic() - start
    return {
        "id": entry_id,
        "name": name,
        "phase": phase,
        "failure_policy": policy,
        "command": command,
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 3),
        "status": "passed" if result.returncode == 0 else "failed",
    }


def _has_junitxml(command):
    words = shlex.split(command)
    return any(
        word.startswith("--junitxml") or word.startswith("--junit-xml")
        for word in words
    )


def write_summary(results, success, out_dir):
    summary = {
        "success": success,
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "entries": results,
    }

    # Detect all-skipped: every entry is skipped (none passed, none failed).
    # This happens when a fail-fast preflight aborts the entire suite.
    if summary["total"] > 0 and summary["skipped"] == summary["total"]:
        summary["all_skipped"] = True
    else:
        summary["all_skipped"] = False

    (out_dir / "integration-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gha = os.environ.get("GITHUB_STEP_SUMMARY")
    if gha:
        with open(gha, "a", encoding="utf-8") as fh:
            fh.write("## Integration test results\n\n")
            fh.write(f"- Total: {summary['total']}\n")
            fh.write(f"- Passed: {summary['passed']}\n")
            fh.write(f"- Failed: {summary['failed']}\n")
            fh.write(f"- Skipped: {summary['skipped']}\n")
            if summary["all_skipped"]:
                fh.write("\n**WARNING: All integration tests were skipped** (preflight abort).\n")
            fh.write("\n")
            if summary["failed"]:
                fh.write("| Entry | Phase | Exit | Duration (s) |\n")
                fh.write("| --- | --- | ---: | ---: |\n")
                for r in results:
                    if r["status"] == "failed":
                        fh.write(
                            f"| {r['name']} | {r['phase']} | {r['exit_code']} "
                            f"| {r['duration_seconds']} |\n"
                        )
                fh.write("\n")


def main():
    args = parse_args()
    tests, environment = load_configuration(args.allow_empty)
    if args.validate_only:
        print(f"Validated {len(tests)} configured integration test(s)")
        return 0

    command_environment = os.environ.copy()
    command_environment.update(environment)
    workdir = os.environ.get("INTEGRATION_WORKDIR")
    if workdir and not os.path.isdir(workdir):
        raise SystemExit(f"INTEGRATION_WORKDIR does not exist: {workdir}")
    if not workdir:
        raise SystemExit("INTEGRATION_WORKDIR is not set")

    report_root = Path(workdir) / REPORT_DIR
    os.makedirs(report_root, exist_ok=True)

    results = []
    aborted = False
    for index, test in enumerate(tests, start=1):
        entry = run_entry(
            test, command_environment, workdir, report_root, index, len(tests)
        )
        results.append(entry)
        if entry["status"] == "failed":
            if entry["failure_policy"] == "fail-fast":
                print(
                    f"Integration test '{entry['name']}' failed with exit code "
                    f"{entry['exit_code']}; policy=fail-fast, aborting remaining entries",
                    flush=True,
                )
                aborted = True
                break
            print(
                f"Integration test '{entry['name']}' failed with exit code "
                f"{entry['exit_code']}; policy=continue-after-failure",
                flush=True,
            )

    success = not aborted and all(r["status"] == "passed" for r in results)
    if aborted:
        # Mark the unrun entries as skipped so the summary reflects the abort.
        for index, test in enumerate(tests[index:], start=index + 1):
            results.append(
                {
                    "id": _entry_id(test, index),
                    "name": test["name"],
                    "phase": _entry_phase(test, index),
                    "failure_policy": _entry_failure_policy(test, index),
                    "command": test["command"],
                    "exit_code": None,
                    "duration_seconds": 0.0,
                    "status": "skipped",
                }
            )

    write_summary(results, success, report_root)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
