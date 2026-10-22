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
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import signal
import threading
import time
import shlex
import subprocess
import sys
from pathlib import Path


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
_SUMMARY_FAILURE_RE = re.compile(r"\b(?:failed|errors?|xfailed|xpassed|no tests ran)\b")

# Signals that mean the process died rather than exited: SIGSEGV from a bad
# address in a teardown destructor is the observed case; the rest are the same
# class.
_FATAL_SIGNALS = {signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL}

_TEARDOWN_CRASH_ENV = "INTEGRATION_ALLOW_TEARDOWN_CRASH"
_EXECUTION_MODES = {"parallel", "exclusive"}
_DEFAULT_MAX_PARALLEL = 2

# Group commands run in their own sessions, so a cancelled runner does not
# reach them through its process group. Shared batches run commands on worker
# threads, where KeyboardInterrupt never arrives; the main thread kills every
# registered command instead, and a command started after cancellation is
# killed as soon as it registers.
_ACTIVE_PROCESSES = set()
_ACTIVE_LOCK = threading.Lock()
_CANCELLED = threading.Event()
# Serializes ledger record updates with the ledger's JSON snapshot.
_LEDGER_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


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
        execution = test.get("execution")
        if execution is not None and execution not in _EXECUTION_MODES:
            raise SystemExit(
                f"integration_tests[{index}].execution must be one of "
                f"{', '.join(sorted(_EXECUTION_MODES))}"
            )
        validate_environment(
            test_environment,
            f"integration_tests[{index}].environment",
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
            return line.strip("= ")
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


def required_mpirun(command, test_environment):
    """Refuse a command whose MPI launcher is absent before bash starts it.

    Without this check, the command starts, ``mpirun`` prints
    ``command not found``, and the following pytest receives SIGPIPE while
    writing its collection report. The runner then reports a crash, hiding
    the missing launcher that actually stopped the group.
    """
    try:
        words = shlex.split(command)
    except ValueError:
        return
    mpirun = next((word for word in words if Path(word).name == "mpirun"), None)
    if mpirun is None:
        return
    if Path(mpirun).is_absolute() or len(Path(mpirun).parts) > 1:
        if Path(mpirun).is_file() and os.access(mpirun, os.X_OK):
            return
    else:
        for directory in test_environment.get("PATH", os.defpath).split(os.pathsep):
            if directory and (Path(directory) / "mpirun").is_file():
                if os.access(Path(directory) / "mpirun", os.X_OK):
                    return
                break
    raise SystemExit(
        "mpirun is required by this integration group but is not installed or on PATH"
    )


def run_command(command, test_environment, workdir, output_prefix="", output_lock=None):
    """Run one group command, streaming its output live while capturing it.

    Streaming (not print-at-end) keeps long hardware groups observable and
    leaves the log behind if the runner itself is killed; the captured copy
    feeds the teardown-crash check.
    """
    required_mpirun(command, test_environment)
    timed_out = threading.Event()
    process = subprocess.Popen(
        ["bash", "-c", f"set -euo pipefail\n{command}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=test_environment,
        cwd=workdir,
        start_new_session=True,
    )
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.add(process)
        cancelled = _CANCELLED.is_set()
    if cancelled:
        kill_process_group(process)

    def expire():
        timed_out.set()
        kill_process_group(process)

    timer = threading.Timer(
        float(test_environment.get("FLAGCICD_CASE_TIMEOUT", "1800")), expire
    )
    timer.start()
    assert process.stdout is not None
    try:
        lines = []
        for line in process.stdout:
            # Shared groups interleave whole, prefixed lines; a lone group's
            # output is written untouched.
            with output_lock or contextlib.nullcontext():
                sys.stdout.write(output_prefix + line)
                if output_lock or output_prefix:
                    sys.stdout.flush()
            lines.append(line)
        sys.stdout.flush()
        returncode = process.wait()
    except KeyboardInterrupt:
        # A cancelled runner must not leave a pytest process writing to the
        # report directory or to a retry's workspace.
        kill_process_group(process)
        process.wait()
        raise
    except (OSError, ValueError):
        # The timeout/cancellation path closes the parent's pipe explicitly;
        # a descendant that escaped with setsid() must not keep this reader
        # waiting for EOF.  The process return code still determines the
        # group's verdict below.
        returncode = process.wait()
    finally:
        timer.cancel()
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(process)
    return (124 if timed_out.is_set() else returncode), "".join(lines)


def kill_process_group(process):
    descendants = set(descendant_pids(process.pid))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    # A vendor launcher can call setsid() and leave the process group.  Kill
    # the descendants captured on both sides of the group kill as well,
    # otherwise one escaped child can keep the report pipe open after timeout.
    descendants.update(descendant_pids(process.pid))
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    stream = process.stdout
    if stream and not stream.closed:
        stream.close()


def descendant_pids(root_pid):
    """Return descendants using procfs, with a portable ps fallback."""
    children = {}
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for task in proc_root.glob("*/task/*/children"):
            try:
                parent = int(task.parts[2])
                values = task.read_text().split()
                children.setdefault(parent, []).extend(int(value) for value in values)
            except (OSError, ValueError):
                continue
    else:
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,ppid="],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            result = None
        if result and result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) == 2:
                    try:
                        children.setdefault(int(fields[1]), []).append(int(fields[0]))
                    except ValueError:
                        continue
    found = []
    pending = list(children.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.append(pid)
        pending.extend(children.get(pid, ()))
    return found


def kill_active_processes():
    """Stop every running group command and refuse to start new ones."""
    with _ACTIVE_LOCK:
        _CANCELLED.set()
        processes = list(_ACTIVE_PROCESSES)
    for process in processes:
        kill_process_group(process)


def cancel_batch_processes():
    """Stop only the commands in the failed shared batch."""
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for process in processes:
        kill_process_group(process)


def max_parallel_workers(environment):
    """Bound on concurrently running shared groups; 1 restores a serial run."""
    value = environment.get("FLAGCICD_MAX_PARALLEL", str(_DEFAULT_MAX_PARALLEL))
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise SystemExit("FLAGCICD_MAX_PARALLEL must be a positive integer")
    return int(value)


def execution_mode(test):
    """Return the declared mode; ``sequential`` marks a legacy, unplanned run."""
    return test.get("execution", "sequential")


def shareable(test):
    """Only explicitly shared single-accelerator L2/L3 groups may overlap."""
    resources = test.get("resources")
    return (
        execution_mode(test) == "parallel"
        and test.get("level") in ("L2", "L3")
        and isinstance(resources, dict)
        and resources.get("min_accelerators") == 1
        and resources.get("shared") is True
    )


def execution_plan(tests):
    """Split the manifest into ordered steps of ``(shared, indexes)``.

    Consecutive shareable groups of one level form a shared batch; every
    other group is its own step. An exclusive group therefore never overlaps
    anything, the manifest's level order (L2, L3, L4) is kept, and so is the
    order of groups across an exclusive boundary.
    """
    plan = []
    for index, test in enumerate(tests):
        shared = shareable(test)
        if (
            shared
            and plan
            and plan[-1][0]
            and tests[plan[-1][1][-1]].get("level") == test.get("level")
        ):
            plan[-1][1].append(index)
        else:
            plan.append((shared, [index]))
    return plan


def run_batch(indexes, workers, run_group):
    """Run one shared batch and return the first observed failure.

    A failure stops further groups of the batch from starting and terminates
    commands already running in that batch. Their terminal records remain in
    the ledger, while groups that have not started are recorded as ``not_run``.
    Ctrl-C/cancellation kills every running group before re-raising.
    """
    failures = {}
    first_failure = None
    stop = threading.Event()

    def guarded(index):
        if stop.is_set() or _CANCELLED.is_set():
            return None
        status = run_group(index)
        if status is not None:
            stop.set()
            cancel_batch_processes()
        return status

    executor = ThreadPoolExecutor(max_workers=min(workers, len(indexes)))
    try:
        futures = {executor.submit(guarded, index): index for index in indexes}
        for future in as_completed(futures):
            status = future.result()
            if status is not None:
                failures[futures[future]] = status
                # A sibling killed by fail-fast reports SIGKILL.  Do not let
                # that cancellation mask the group that caused the batch to
                # stop if its future completes first.
                if first_failure is None or (
                    first_failure == -signal.SIGKILL and status != -signal.SIGKILL
                ):
                    first_failure = status
    except BaseException:
        stop.set()
        kill_active_processes()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return first_failure


def resource_block(test, environment):
    """Check recorded hardware before running a group; never skip it as a pass."""
    if test.get("blocked_reason"):
        return test["blocked_reason"]
    if "metrics_directory" in test.get("requires", []):
        metrics = environment.get("FLAGCICD_METRICS_DIR", "")
        if not metrics or not Path(metrics).is_dir():
            return "metrics directory is unavailable: FLAGCICD_METRICS_DIR"
    resources = test.get("resources")
    if resources is None:
        return ""
    minimum = resources.get("min_accelerators") if isinstance(resources, dict) else None
    if type(minimum) is not int or minimum < 1:
        return "invalid accelerator resource requirement"
    platform = test.get("platform", "")
    if not isinstance(platform, str) or not re.fullmatch(r"[a-z0-9_-]+", platform):
        return "resource check requires a platform identity"
    try:
        folder = Path(environment["FLAGCICD_REPORT_DIR"])
        hardware = json.loads((folder / f"hardware-{platform}.json").read_text())
        count = hardware["accelerator_count"]
        if type(count) is not int or count < 0:
            raise ValueError("invalid accelerator count")
        if "primary_device" in test.get("requires", []):
            current = hardware.get("current_device")
            if type(current) is not int:
                raise ValueError("current device evidence is unavailable")
            if current != 0:
                return (
                    f"requires primary device 0; recorded current device is {current}"
                )
    except (KeyError, OSError, ValueError, TypeError) as error:
        return f"hardware resource evidence is unavailable: {error}"
    if count < minimum:
        return f"requires {minimum} accelerators; recorded hardware has {count}"
    return ""


def make_record(test):
    """Create the stable, ordered ledger entry for one configured group."""
    return {
        "name": test["name"],
        "id": test.get("id", test["name"]),
        "level": test.get("level", "L2"),
        "category": test.get("category", "functional"),
        "command": test["command"],
        "requires_pytest": "pytest" in test["command"]
        or "run_unit_tests.py" in test["command"],
        "status": "pending",
        "execution": execution_mode(test),
        "min_passed": test.get("min_passed", 0),
        "requires": test.get("requires", []),
        "resources": test.get("resources", {}),
        "selection_reason": test.get("selection_reason", "configured_group"),
        "log_url": (
            os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            + "/"
            + os.environ.get("GITHUB_REPOSITORY", "")
            + "/actions/runs/"
            + os.environ.get("GITHUB_RUN_ID", "")
        )
        if os.environ.get("GITHUB_RUN_ID")
        else "",
    }


def prepare_test_environment(test, command_environment):
    """Apply group-specific variables without mutating the shared environment."""
    test_environment = command_environment.copy()
    for key, value in test.get("environment", {}).items():
        test_environment[key] = os.path.expandvars(value)
    if test_environment.get("FLAGCICD_REPORT_DIR"):
        test_environment["FLAGCICD_TEST_GROUP"] = test["name"]
    test_environment["FLAGCICD_CASE_TIMEOUT"] = str(test.get("timeout_seconds", 1800))
    return test_environment


def execute_group(
    test, record, banner, command_environment, workdir, save, output_lock=None
):
    """Execute one group, recording its outcome in its ledger entry.

    Returns ``None`` on success, otherwise the ``SystemExit`` status for the
    failure: the command's return code, or a message when the group could not
    start. Shared batches call this from worker threads, so every ledger
    update holds ``_LEDGER_LOCK`` and only an ``output_lock`` caller gets the
    per-line group prefix; serial output stays byte-identical, which keeps
    workflow commands such as ``::error::`` working.
    """

    def report(message):
        with output_lock or contextlib.nullcontext():
            print(message, flush=True)

    def finish(status, returncode, reason=""):
        finished = time.time()
        with _LEDGER_LOCK:
            record.update(
                status=status,
                returncode=returncode,
                finished=finished,
                duration_seconds=finished - record["started"],
            )
            if reason:
                record["reason"] = reason
        save()

    report(f"===== {banner} =====")
    with _LEDGER_LOCK:
        record.update(status="running", started=time.time())
    save()
    test_environment = prepare_test_environment(test, command_environment)
    reason = resource_block(test, test_environment)
    if reason:
        finish("blocked", 1, reason)
        return f"Integration group {test['name']!r} is blocked: {reason}"
    allow_crash = (
        teardown_crash_allowed(test_environment) if not test.get("level") else False
    )
    try:
        returncode, output = run_command(
            test["command"],
            test_environment,
            workdir,
            output_prefix=(
                f"[{test['name']}] " if output_lock or test.get("stream_output") else ""
            ),
            output_lock=output_lock,
        )
    except (OSError, SystemExit) as error:
        finish("failed", 1, str(error))
        return error.code if isinstance(error, SystemExit) else str(error)
    ok, tolerated = classify_command_result(returncode, output, allow_crash)
    finish(
        "passed" if ok else ("timeout" if returncode == 124 else "failed"), returncode
    )
    if tolerated:
        report(
            f"Integration test '{test['name']}' passed ({clean_summary(output)}), "
            f"then the command died with signal {signal_number(returncode)} during "
            f"teardown ({_TEARDOWN_CRASH_ENV} is set for this group; see this "
            "script's header comment)"
        )
    if ok:
        return None
    report(f"Integration test '{test['name']}' failed with exit code {returncode}")
    return returncode


def mark_not_run(records, reason):
    """Make unstarted work visible without claiming it passed."""
    now = time.time()
    with _LEDGER_LOCK:
        for record in records:
            if record["status"] == "pending":
                record.update(
                    status="not_run",
                    reason=reason,
                    returncode=1,
                    started=now,
                    finished=now,
                    duration_seconds=0.0,
                )


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
    ledger_path = (
        Path(os.environ["FLAGCICD_REPORT_DIR"]) / "groups.json"
        if os.environ.get("FLAGCICD_REPORT_DIR")
        else None
    )
    ledger = {
        "expected": [test["name"] for test in tests],
        "groups": [],
        "complete": False,
    }

    def save():
        if ledger_path:
            with _LEDGER_LOCK:
                text = json.dumps(ledger, indent=2) + "\n"
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                ledger_path.write_text(text)

    def banner(index):
        return f"[{index + 1}/{len(tests)}] {tests[index]['name']}"

    # Test fixtures and external callers that predate execution modes keep the
    # original fail-fast serial run, whose ledger grows as groups start. The
    # tier selector gives every composed group a mode, so CI takes the
    # planned path below.
    if not any("execution" in test for test in tests):
        for index, test in enumerate(tests):
            record = make_record(test)
            ledger["groups"].append(record)
            status = execute_group(
                test, record, banner(index), command_environment, workdir, save
            )
            if status is not None:
                raise SystemExit(status)
        ledger["complete"] = True
        save()
        return 0

    # Every planned group is in the ledger, in manifest order, before anything
    # starts; a failure turns the groups that never started into `not_run`.
    records = [make_record(test) for test in tests]
    ledger["groups"] = records
    save()
    workers = max_parallel_workers(command_environment)
    output_lock = threading.Lock()

    def run_group(index):
        return execute_group(
            tests[index],
            records[index],
            banner(index),
            command_environment,
            workdir,
            save,
            output_lock,
        )

    for shared, indexes in execution_plan(tests):
        if shared and len(indexes) > 1:
            print(
                f"===== {tests[indexes[0]]['level']} shared batch: "
                f"{len(indexes)} group(s), at most {workers} at a time =====",
                flush=True,
            )
            status = run_batch(indexes, workers, run_group)
        else:
            index = indexes[0]
            status = execute_group(
                tests[index],
                records[index],
                banner(index),
                command_environment,
                workdir,
                save,
            )
        if status is not None:
            mark_not_run(records, "not_run_after_failure")
            save()
            raise SystemExit(status)
    ledger["complete"] = True
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
