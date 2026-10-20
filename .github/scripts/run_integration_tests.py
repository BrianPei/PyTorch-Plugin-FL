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

"""Run the integration groups one platform's plan selected.

The list comes from the composer, filtered by the plan: each entry carries the
manifest id it came from, the group's name, its command, its environment, and
optionally the seconds it may run and the ids it depends on.

Every group's outcome is written to one JSON file so the platform report can
compare what ran against what the plan said would run. The alternative -- a
log a human searches -- cannot distinguish "this group passed" from "this
group was never selected", which is exactly the distinction a gate has to make.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time


# A group command can die with a fatal signal *after* pytest has finished:
# some vendor containers tear their driver stack down during exit() (an ELF
# destructor faults), so the process dies with SIGSEGV although every test ran
# and reported. INTEGRATION_ALLOW_TEARDOWN_CRASH (set per group through that
# group's `environment` mapping) reads that case -- a complete, clean pytest
# summary followed by a fatal signal -- as a pass and prints it as a warning.
# Without the variable the crash is a failure, which is the right reading for
# a signal that arrives before the summary: nothing proved the tests
# themselves were fine. This mirrors UNIT_TESTS_ALLOW_TEARDOWN_CRASH in
# run_unit_tests.py; the two scripts are standalone files with no shared
# package, so the small matching logic below is duplicated on purpose -- keep
# the pair in sync by hand.
#
# The output below is both streamed live (a multi-minute hardware group must
# stay observable, and a killed runner must leave its log behind) and
# accumulated for the summary check.
_SUMMARY_RE = re.compile(r"\bin \d+(?:\.\d+)?s\b")
_SUMMARY_FAILURE_RE = re.compile(r"\b(failed|error|errors|no tests ran)\b")

# Signals that mean the process died rather than exited: SIGSEGV from a bad
# address in a teardown destructor is the observed case; the rest are the same
# class.
_FATAL_SIGNALS = {signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL}

_TEARDOWN_CRASH_ENV = "INTEGRATION_ALLOW_TEARDOWN_CRASH"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--groups-output",
        default="",
        help="write one JSON record per selected group to this path",
    )
    parser.add_argument(
        "--group-log-dir",
        default="",
        help="retain each group's captured output here",
    )
    return parser.parse_args()


def _single_line(value):
    return isinstance(value, str) and bool(value.strip()) and "\n" not in value


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
        test_environment = test.get("environment", {})
        if not _single_line(name) or "\r" in name:
            raise SystemExit(
                f"integration_tests[{index}].name must be a non-empty single-line string"
            )
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise SystemExit(
                f"integration_tests[{index}].command must be a non-empty string"
            )
        validate_environment(
            test_environment,
            f"integration_tests[{index}].environment",
        )
        timeout = test.get("timeout_seconds")
        if timeout is not None and (
            not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1
        ):
            raise SystemExit(
                f"integration_tests[{index}].timeout_seconds must be a positive integer"
            )
        requires = test.get("requires", [])
        if not isinstance(requires, list) or not all(
            _single_line(item) for item in requires
        ):
            raise SystemExit(
                f"integration_tests[{index}].requires must be a list of group ids"
            )
        group_id = test.get("id")
        if group_id is not None and not _single_line(group_id):
            raise SystemExit(
                f"integration_tests[{index}].id must be a non-empty single-line string"
            )

    validate_environment(environment, "integration_environment")

    return tests, environment


def validate_environment(environment, field_name):
    """Validate one integration-test environment mapping."""
    if not isinstance(environment, dict):
        raise SystemExit(f"{field_name} must be an object")
    for key, value in environment.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(
                f"invalid environment variable name in {field_name}: {key}"
            )
        if (
            not isinstance(value, str)
            or "\n" in value
            or "\r" in value
            or "\0" in value
        ):
            raise SystemExit(f"{field_name}[{key}] must be a single-line string")


def clean_summary(output):
    """Return pytest's summary line when it shows a completed, clean run.

    ``None`` means no such line exists (the command died before the session
    ended) or the summary reports failures -- either way the group's verdict
    is unknown or negative, never a pass.
    """
    for line in reversed(output.splitlines()):
        if _SUMMARY_RE.search(line):
            if _SUMMARY_FAILURE_RE.search(line):
                return None
            return line.strip(" =")
    return None


def teardown_crash_allowed(test_environment):
    """Whether this group tolerates a fatal signal after a clean summary."""
    return test_environment.get(_TEARDOWN_CRASH_ENV, "").lower() not in (
        "",
        "0",
        "off",
        "false",
    )


def classify_command_result(returncode, output, allow_teardown_crash):
    """Map one group command onto (ok, tolerated).

    Only a fatal *signal* after a complete, clean pytest summary is
    tolerable, and only when the group opted in: a plain nonzero exit, a
    signal before the summary, or a summary reporting failures all fail, with
    or without the opt-in.
    """
    if returncode == 0:
        return True, False
    fatal = (returncode < 0 and -returncode in _FATAL_SIGNALS) or (
        returncode > 128 and (returncode - 128) in _FATAL_SIGNALS
    )
    if fatal and allow_teardown_crash and clean_summary(output):
        return True, True
    return False, False


def signal_number(returncode):
    """Human-readable signal for a fatal-signal returncode."""
    if returncode < 0:
        return -returncode
    return returncode - 128


def run_command(command, test_environment, workdir, timeout_seconds=None):
    """Run one group command, streaming its output live while capturing it.

    Streaming (not print-at-end) keeps long hardware groups observable and
    leaves the log behind if the runner itself is killed; the captured copy
    feeds the teardown-crash check.

    A timeout kills the process *group*, because the group's command is a
    shell that may have started pytest and its workers: signalling only the
    shell would leave the hardware busy for the next group and report a
    timeout that nothing enforced.
    """
    process = subprocess.Popen(
        ["bash", "-c", f"set -euo pipefail\n{command}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=test_environment,
        cwd=workdir,
        start_new_session=True,
    )
    assert process.stdout is not None
    expired = threading.Event()

    def expire():
        expired.set()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = None
    if timeout_seconds:
        timer = threading.Timer(timeout_seconds, expire)
        timer.daemon = True
        timer.start()
    lines = []
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            lines.append(line)
        sys.stdout.flush()
        returncode = process.wait()
    finally:
        if timer is not None:
            timer.cancel()
    return returncode, "".join(lines), expired.is_set()


def group_record(test, status, returncode=None, duration=0.0, detail="", log=""):
    return {
        "id": test.get("id") or test["name"],
        "name": test["name"],
        "status": status,
        "exit_code": returncode,
        "duration_seconds": round(duration, 3),
        "detail": detail,
        "log": log,
    }


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
    log_dir = args.group_log_dir
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    records, status_by_id = [], {}
    failure = 0
    for index, test in enumerate(tests, start=1):
        group_id = test.get("id") or test["name"]
        if failure:
            records.append(
                group_record(test, "blocked", detail="an earlier group failed")
            )
            continue
        unmet = [
            requirement
            for requirement in test.get("requires", [])
            if status_by_id.get(requirement) != "passed"
        ]
        if unmet:
            detail = f"dependency-blocked: {', '.join(unmet)} did not pass"
            print(
                f"===== [{index}/{len(tests)}] {test['name']}: {detail} =====",
                flush=True,
            )
            records.append(group_record(test, "blocked", detail=detail))
            status_by_id[group_id] = "blocked"
            continue

        print(
            f"===== [{index}/{len(tests)}] {test['name']} =====",
            flush=True,
        )
        test_environment = command_environment.copy()
        for key, value in test.get("environment", {}).items():
            test_environment[key] = os.path.expandvars(value)
        if test_environment.get("FLAGCICD_REPORT_DIR"):
            test_environment["FLAGCICD_TEST_GROUP"] = test["name"]
        allow_crash = teardown_crash_allowed(test_environment)
        started = time.monotonic()
        returncode, output, timed_out = run_command(
            test["command"],
            test_environment,
            workdir,
            test.get("timeout_seconds"),
        )
        duration = time.monotonic() - started
        log_path = ""
        if log_dir:
            log_path = os.path.join(log_dir, f"{index:02d}-{group_id}.log")
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(output)
        if timed_out:
            print(
                f"Integration test '{test['name']}' exceeded "
                f"{test['timeout_seconds']}s and was killed",
                flush=True,
            )
            records.append(
                group_record(
                    test,
                    "timed_out",
                    returncode,
                    duration,
                    f"exceeded {test['timeout_seconds']}s",
                    log_path,
                )
            )
            status_by_id[group_id] = "timed_out"
            failure = returncode or 1
            continue
        ok, tolerated = classify_command_result(returncode, output, allow_crash)
        if tolerated:
            print(
                f"Integration test '{test['name']}' passed "
                f"({clean_summary(output)}), then the command died with "
                f"signal {signal_number(returncode)} during teardown "
                f"({_TEARDOWN_CRASH_ENV} is set for this group; see this "
                f"script's header comment)",
                flush=True,
            )
        if ok:
            records.append(
                group_record(
                    test,
                    "passed",
                    returncode,
                    duration,
                    "tolerated teardown crash" if tolerated else "",
                    log_path,
                )
            )
            status_by_id[group_id] = "passed"
            continue
        print(
            f"Integration test '{test['name']}' failed with exit code {returncode}",
            flush=True,
        )
        records.append(
            group_record(test, "failed", returncode, duration, "nonzero exit", log_path)
        )
        status_by_id[group_id] = "failed"
        failure = returncode

    if args.groups_output:
        with open(args.groups_output, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "schema_version": "flagcicd.groups/v1",
                    "platform": os.environ.get("FLAGCICD_PLATFORM", ""),
                    "groups": records,
                },
                output,
                indent=2,
            )
            output.write("\n")
    return failure


if __name__ == "__main__":
    raise SystemExit(main())
