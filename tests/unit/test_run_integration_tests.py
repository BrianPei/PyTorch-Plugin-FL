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

"""Tests for the data-driven integration test runner."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / ".github/scripts/run_integration_tests.py"
)
_SPEC = importlib.util.spec_from_file_location("run_integration_tests", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def test_per_test_environment_is_expanded_and_scoped(monkeypatch, tmp_path):
    tests = [
        {
            "name": "profiled",
            "environment": {"LD_PRELOAD": "${ASCEND_MSPTI_PRELOAD}"},
            "command": (
                "printf '%s' \"${LD_PRELOAD-<unset>}\" > profiled.txt; "
                "printf '%s' \"$SHARED\" > shared.txt"
            ),
        },
        {
            "name": "ordinary",
            "command": "printf '%s' \"${LD_PRELOAD-<unset>}\" > ordinary.txt",
        },
    ]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({"SHARED": "value"}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ASCEND_MSPTI_PRELOAD", "/opt/cann/libmspti.so:/opt/base.so")
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    class Args:
        validate_only = False
        allow_empty = False

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())

    assert _RUNNER.main() == 0
    assert (tmp_path / "profiled.txt").read_text() == (
        "/opt/cann/libmspti.so:/opt/base.so"
    )
    assert (tmp_path / "ordinary.txt").read_text() == "<unset>"
    assert (tmp_path / "shared.txt").read_text() == "value"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ([], "must be an object"),
        ({"NOT-VALID": "value"}, "invalid environment variable name"),
        ({"VALID": "line one\nline two"}, "must be a single-line string"),
    ],
)
def test_per_test_environment_validation(monkeypatch, environment, message):
    tests = [{"name": "test", "command": "true", "environment": environment}]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")

    with pytest.raises(SystemExit, match=message):
        _RUNNER.load_configuration(allow_empty=False)


def test_execution_mode_validation(monkeypatch):
    tests = [{"name": "test", "command": "true", "execution": "overlap"}]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")

    with pytest.raises(SystemExit, match="execution must be one of"):
        _RUNNER.load_configuration(allow_empty=False)


def _run_main(monkeypatch, tmp_path, tests):
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))

    class Args:
        validate_only = False
        allow_empty = False

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())
    return _RUNNER.main()


CLEAN_SUMMARY_COMMAND = "printf '2 passed in 0.01s\\n'; kill -SEGV $$"
BANNERED_CLEAN_SUMMARY_COMMAND = (
    "printf '===== 2 passed, 1 skipped in 0.01s =====\\n'; kill -SEGV $$"
)
OPT_IN = {"INTEGRATION_ALLOW_TEARDOWN_CRASH": "1"}


def test_summary_line_is_recognised_with_a_banner_and_without_one():
    assert _RUNNER.clean_summary("2 passed in 0.01s") == "2 passed in 0.01s"
    assert (
        _RUNNER.clean_summary("===== 2 passed, 1 skipped in 0.01s =====")
        == "2 passed, 1 skipped in 0.01s"
    )
    assert _RUNNER.clean_summary("2 passed, 1 xfailed in 0.01s") is None


def test_missing_mpirun_is_caught_before_the_group_starts(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    tests = [
        {"name": "mpi", "command": "mpirun -np 2 python -m pytest"},
        {"name": "next", "command": "touch should_not_run"},
    ]

    with pytest.raises(SystemExit, match="mpirun is required"):
        _run_main(monkeypatch, tmp_path, tests)

    assert not (tmp_path / "should_not_run").exists()


def test_teardown_crash_after_clean_summary_is_tolerated_when_opted_in(
    monkeypatch, tmp_path
):
    """A fatal signal after pytest's clean summary passes with the opt-in.

    Mirrors the Ascend profiler group: the suite reports green, then a vendor
    destructor faults during interpreter teardown.
    """
    tests = [
        {
            "name": "teardown-crash",
            "command": CLEAN_SUMMARY_COMMAND,
            "environment": dict(OPT_IN),
        }
    ]
    assert _run_main(monkeypatch, tmp_path, tests) == 0


def test_teardown_crash_after_a_bannered_summary_is_tolerated(monkeypatch, tmp_path):
    tests = [
        {
            "name": "bannered-teardown-crash",
            "command": BANNERED_CLEAN_SUMMARY_COMMAND,
            "environment": dict(OPT_IN),
        }
    ]
    assert _run_main(monkeypatch, tmp_path, tests) == 0


def test_teardown_crash_without_opt_in_still_fails(monkeypatch, tmp_path):
    tests = [{"name": "teardown-crash", "command": CLEAN_SUMMARY_COMMAND}]
    with pytest.raises(SystemExit) as error:
        _run_main(monkeypatch, tmp_path, tests)
    assert error.value.code == -signal.SIGSEGV


def test_ordinary_failure_stops_the_run(monkeypatch, tmp_path):
    tests = [
        {"name": "first", "command": "exit 5"},
        {"name": "second", "command": "touch should_not_run"},
    ]

    with pytest.raises(SystemExit) as error:
        _run_main(monkeypatch, tmp_path, tests)

    assert error.value.code == 5
    assert not (tmp_path / "should_not_run").exists()


def test_failure_summary_is_not_tolerated_even_when_opted_in(monkeypatch, tmp_path):
    """A summary reporting failures fails with or without the opt-in: only a
    *clean* summary proves the tests passed before the signal."""
    tests = [
        {
            "name": "failed-suite",
            "command": "printf '1 failed, 1 passed in 0.02s\\n'; exit 1",
            "environment": dict(OPT_IN),
        }
    ]
    with pytest.raises(SystemExit) as error:
        _run_main(monkeypatch, tmp_path, tests)
    assert error.value.code == 1


@pytest.mark.parametrize(
    ("returncode", "output", "allow", "expected"),
    [
        (0, "", False, (True, False)),
        (-signal.SIGSEGV, "2 passed in 0.01s\n", True, (True, True)),
        (-signal.SIGSEGV, "2 passed in 0.01s\n", False, (False, False)),
        # Signal before any summary: nothing proved the tests passed.
        (-signal.SIGSEGV, "some traceback\n", True, (False, False)),
        # Plain nonzero exit, even with a clean summary, is a failure: only
        # fatal *signals* are tolerable, never exit statuses.
        (1, "2 passed in 0.01s\n", True, (False, False)),
        # A runner kill (SIGKILL, e.g. OOM or job timeout) must never be
        # forgiven: it is not a teardown destructor fault.
        (-signal.SIGKILL, "2 passed in 0.01s\n", True, (False, False)),
        # Shell-encoded signal (128+N) reads the same as negative-N.
        (128 + signal.SIGSEGV, "2 passed in 0.01s\n", True, (True, True)),
    ],
)
def test_classify_command_result(returncode, output, allow, expected):
    assert _RUNNER.classify_command_result(returncode, output, allow) == expected


@pytest.mark.parametrize(
    "hardware,minimum,blocked",
    [(1, 2, True), (2, 2, False), (None, 1, True), (True, 1, True)],
)
def test_resource_requirements_use_recorded_hardware_and_keep_a_ledger(
    monkeypatch, tmp_path, hardware, minimum, blocked
):
    reports = tmp_path / "reports"
    reports.mkdir()
    if hardware is not None:
        (reports / "hardware-cuda.json").write_text(
            json.dumps({"accelerator_count": hardware})
        )
    command = "printf executed > executed.txt"
    group = {
        "id": "resource-fixture",
        "name": "resource fixture",
        "platform": "cuda",
        "command": command,
        "resources": {"min_accelerators": minimum},
        "requires": ["accelerator"],
    }
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps([group]))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(reports))
    monkeypatch.setattr(
        _RUNNER,
        "parse_args",
        lambda: type("Args", (), {"validate_only": False, "allow_empty": False})(),
    )
    if blocked:
        with pytest.raises(SystemExit, match="blocked"):
            _RUNNER.main()
        assert not (tmp_path / "executed.txt").exists()
    else:
        assert _RUNNER.main() == 0
        assert (tmp_path / "executed.txt").read_text() == "executed"
    ledger = json.loads((reports / "groups.json").read_text())
    record = ledger["groups"][0]
    assert record["status"] == ("blocked" if blocked else "passed")
    assert record["duration_seconds"] >= 0 and record["finished"] >= record["started"]
    assert record["resources"] == {"min_accelerators": minimum}
    assert ledger["complete"] is (not blocked)


def test_acceptance_requirements_need_metrics_and_primary_device_evidence(
    monkeypatch, tmp_path
):
    reports = _reports(monkeypatch, tmp_path)
    environment = {
        "FLAGCICD_REPORT_DIR": str(reports),
        "FLAGCICD_METRICS_DIR": str(tmp_path / "missing-metrics"),
    }
    metrics_group = _group("performance", "true", execution="exclusive")
    metrics_group["requires"] = ["accelerator", "metrics_directory"]
    assert "metrics directory is unavailable" in _RUNNER.resource_block(
        metrics_group, environment
    )

    primary_group = _group("models", "true", execution="exclusive")
    primary_group["requires"] = ["accelerator", "primary_device"]
    (reports / "hardware-cuda.json").write_text(
        json.dumps({"accelerator_count": 1, "current_device": 1})
    )
    assert "primary device 0" in _RUNNER.resource_block(primary_group, environment)


def test_timeout_kills_a_setsid_grandchild(tmp_path):
    """An escaped launcher child must not survive the group timeout."""
    process_tree_available = Path("/proc").is_dir()
    if not process_tree_available:
        try:
            process_tree_available = (
                subprocess.run(
                    ["ps", "-eo", "pid=,ppid="],
                    capture_output=True,
                    text=True,
                    check=False,
                ).returncode
                == 0
            )
        except OSError:
            process_tree_available = False
    if not process_tree_available:
        pytest.skip("the host does not expose a process-tree inspection API")
    child = tmp_path / "child.py"
    marker = tmp_path / "escaped-child-survived"
    child.write_text(
        "import pathlib, time\n"
        "time.sleep(0.5)\n"
        f"pathlib.Path({str(marker)!r}).write_text('survived')\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, sys.argv[1]], start_new_session=True)\n"
        "time.sleep(30)\n"
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "hardware-cuda.json").write_text(
        json.dumps({"accelerator_count": 1, "current_device": 0})
    )
    group = _group(
        "escaped-launcher",
        f"{sys.executable} {parent} {child}",
        execution="exclusive",
    )
    group["timeout_seconds"] = 0.1
    env = {
        **os.environ,
        "INTEGRATION_TESTS": json.dumps([group]),
        "INTEGRATION_WORKDIR": str(tmp_path),
        "FLAGCICD_REPORT_DIR": str(reports),
    }
    result = subprocess.run(
        [sys.executable, str(_RUNNER_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 124
    time.sleep(0.8)
    assert not marker.exists()


def test_declared_parallel_groups_start_together_and_keep_ledger_order(
    monkeypatch, tmp_path
):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "hardware-cuda.json").write_text(json.dumps({"accelerator_count": 1}))
    tests = [
        {
            "name": "first shared group",
            "command": (
                "touch first.started; "
                "while [[ ! -f second.started ]]; do sleep 0.01; done; "
                "touch first.done"
            ),
            "platform": "cuda",
            "level": "L2",
            "execution": "parallel",
            "resources": {"min_accelerators": 1, "shared": True},
        },
        {
            "name": "second shared group",
            "command": (
                "touch second.started; "
                "while [[ ! -f first.started ]]; do sleep 0.01; done; "
                "touch second.done"
            ),
            "platform": "cuda",
            "level": "L2",
            "execution": "parallel",
            "resources": {"min_accelerators": 1, "shared": True},
        },
    ]
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(reports))
    monkeypatch.setenv("FLAGCICD_MAX_PARALLEL", "2")

    assert _run_main(monkeypatch, tmp_path, tests) == 0
    assert (tmp_path / "first.done").exists()
    assert (tmp_path / "second.done").exists()
    ledger = json.loads((reports / "groups.json").read_text())
    assert ledger["complete"] is True
    assert [group["name"] for group in ledger["groups"]] == [
        "first shared group",
        "second shared group",
    ]
    assert all(group["status"] == "passed" for group in ledger["groups"])
    assert all(group["execution"] == "parallel" for group in ledger["groups"])


def _reports(monkeypatch, tmp_path, accelerators=1):
    """A report directory holding recorded hardware for the cuda fixtures."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "hardware-cuda.json").write_text(
        json.dumps({"accelerator_count": accelerators, "current_device": 0})
    )
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(reports))
    return reports


def _group(name, command, level="L2", execution="parallel", minimum=1):
    return {
        "name": name,
        "command": command,
        "platform": "cuda",
        "level": level,
        "execution": execution,
        "resources": {
            "min_accelerators": minimum,
            **({"shared": True} if execution == "parallel" else {}),
        },
    }


def test_execution_plan_never_overlaps_an_exclusive_or_multi_device_group():
    tests = [
        _group("a", "true"),
        _group("b", "true"),
        _group("compile", "true", execution="exclusive"),
        _group("c", "true"),
        _group("d", "true", level="L3"),
        _group("two-card", "true", level="L3", minimum=2),
        _group("e", "true", level="L3"),
        _group("f", "true", level="L3"),
        _group("soak", "true", level="L4"),
    ]
    assert _RUNNER.execution_plan(tests) == [
        (True, [0, 1]),
        (False, [2]),
        (True, [3]),
        (True, [4]),
        (False, [5]),
        (True, [6, 7]),
        (False, [8]),
    ]


@pytest.mark.parametrize("value", ["0", "-1", "two", "", "1.5"])
def test_invalid_worker_bound_is_rejected(value):
    with pytest.raises(SystemExit, match="FLAGCICD_MAX_PARALLEL"):
        _RUNNER.max_parallel_workers({"FLAGCICD_MAX_PARALLEL": value})


def test_worker_bound_limits_concurrent_shared_groups(monkeypatch, tmp_path):
    # Each group holds a lock directory while it runs; a second holder means
    # two groups overlapped although only one worker was allowed.
    command = (
        "mkdir held || { touch overlapped; exit 0; }; sleep 0.05; rmdir held; "
        "touch {name}.done"
    )
    tests = [_group(name, command.replace("{name}", name)) for name in "abc"]
    _reports(monkeypatch, tmp_path)
    monkeypatch.setenv("FLAGCICD_MAX_PARALLEL", "1")

    assert _run_main(monkeypatch, tmp_path, tests) == 0
    assert not (tmp_path / "overlapped").exists()
    assert all((tmp_path / f"{name}.done").exists() for name in "abc")


def test_shared_failure_cancels_inflight_groups_and_blocks_later_groups(
    monkeypatch, tmp_path, capfd
):
    reports = _reports(monkeypatch, tmp_path)
    tests = [
        _group(
            "slow",
            "touch slow.started; while [[ ! -f fast.failed ]]; do sleep 0.01; done; "
            "sleep 0.05; touch slow.done",
        ),
        _group(
            "fast",
            "while [[ ! -f slow.started ]]; do sleep 0.01; done; "
            "touch fast.failed; exit 7",
        ),
        _group("queued", "touch queued.ran"),
        _group("isolated", "touch isolated.ran", execution="exclusive"),
        _group("later", "touch later.ran", level="L3"),
    ]
    monkeypatch.setenv("FLAGCICD_MAX_PARALLEL", "2")

    with pytest.raises(SystemExit) as error:
        _run_main(monkeypatch, tmp_path, tests)

    assert error.value.code == 7
    # The sibling is terminated as soon as the first failure is observed;
    # nothing new starts after that failure.
    assert not (tmp_path / "slow.done").exists()
    for name in ("queued", "isolated", "later"):
        assert not (tmp_path / f"{name}.ran").exists()
    ledger = json.loads((reports / "groups.json").read_text())
    assert ledger["complete"] is False
    assert [group["name"] for group in ledger["groups"]] == ledger["expected"]
    assert [group["status"] for group in ledger["groups"]] == [
        "failed",
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]
    assert "Integration test 'fast' failed with exit code 7" in capfd.readouterr().out


def test_cancelling_a_shared_batch_kills_every_running_group(tmp_path):
    """SIGINT reaches only the runner; it must still stop worker commands."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "hardware-cuda.json").write_text(json.dumps({"accelerator_count": 1}))
    tests = [
        _group(name, f"touch {name}.started; sleep 30; touch {name}.survived")
        for name in ("a", "b")
    ]
    env = {
        **os.environ,
        "INTEGRATION_TESTS": json.dumps(tests),
        "INTEGRATION_WORKDIR": str(tmp_path),
        "FLAGCICD_REPORT_DIR": str(reports),
        "FLAGCICD_MAX_PARALLEL": "2",
    }
    process = subprocess.Popen(
        [sys.executable, str(_RUNNER_PATH)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while not all((tmp_path / f"{n}.started").exists() for n in "ab"):
        assert time.monotonic() < deadline, "shared groups did not start"
        time.sleep(0.02)
    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=10) != 0
    time.sleep(0.2)
    survivors = subprocess.run(
        ["pgrep", "-f", f"touch {tmp_path}"], capture_output=True, text=True
    )
    assert not (tmp_path / "a.survived").exists()
    assert not survivors.stdout.strip()
    ledger = json.loads((reports / "groups.json").read_text())
    assert ledger["complete"] is False


def test_blocked_shared_group_is_recorded_without_a_pass(monkeypatch, tmp_path):
    reports = _reports(monkeypatch, tmp_path)
    tests = [
        _group("one-card", "touch one.ran"),
        {**_group("blocked", "touch blocked.ran"), "blocked_reason": "no model"},
    ]

    with pytest.raises(SystemExit, match="'blocked' is blocked: no model"):
        _run_main(monkeypatch, tmp_path, tests)

    assert not (tmp_path / "blocked.ran").exists()
    ledger = json.loads((reports / "groups.json").read_text())
    statuses = {group["name"]: group["status"] for group in ledger["groups"]}
    assert statuses["blocked"] == "blocked"
    assert statuses["one-card"] in ("failed", "passed", "not_run")
    assert ledger["complete"] is False
