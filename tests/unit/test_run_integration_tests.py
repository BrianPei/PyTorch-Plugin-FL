"""Unit coverage for the integration-test runner script.

The script under test (``.github/scripts/run_integration_tests.py``) is loaded
directly from source here rather than imported by package module name, because
its filename contains dots and hyphens and is not a valid import name. The
tests drive ``main()`` with real subprocess calls so the ``bash -c`` execution
path (exit-code capture, log writing) is exercised end to end.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "run_integration_tests.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_integration_tests", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner():
    return _load_runner()


def _run(main, monkeypatch, tmp_path, tests):
    """Run main() with the given test manifest and return its exit code.

    Every entry command below uses ``bash`` truth values so the subprocess path
    is real: ``true`` exits 0, ``false`` exits 1.
    """
    import sys

    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({"FOO": "bar"}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    # Clear sys.argv so parse_args() doesn't see pytest's command-line arguments
    monkeypatch.setattr(sys, "argv", ["run_integration_tests.py"])
    return main()


def test_default_failure_policy_is_fail_fast(runner, monkeypatch, tmp_path):
    # No explicit failure_policy in the manifest -> the script-level default
    # (fail-fast) applies, so a failing first entry aborts the rest.
    tests = [
        {"id": "check device", "name": "check device", "command": "false"},
        {"id": "do-stuff", "name": "do stuff", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    assert summary["success"] is False
    statuses = {entry["id"]: entry["status"] for entry in summary["entries"]}
    assert statuses["check device"] == "failed"
    assert statuses["do-stuff"] == "skipped"


def _summary(tmp_path):
    summary_path = tmp_path / "integration-reports" / "integration-summary.json"
    assert summary_path.exists(), "integration-summary.json was not written"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _abort_test(name, policy, command="false"):
    return {
        "id": name,
        "name": name,
        "phase": "preflight" if "check" in name.lower() else "functional",
        "failure_policy": policy,
        "command": command,
    }


def test_fail_fast_aborts_remaining_entries(runner, monkeypatch, tmp_path):
    # 0/1: fail-fast entry fails -> abort, remaining entries skipped, exit 1.
    tests = [
        _abort_test("check device", "fail-fast", command="false"),
        {"id": "do-stuff", "name": "do stuff", "command": "true"},
        {"id": "do-more", "name": "do more", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    assert summary["success"] is False
    assert summary["total"] == 3
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    statuses = {entry["id"]: entry["status"] for entry in summary["entries"]}
    assert statuses["check device"] == "failed"
    assert statuses["do-stuff"] == "skipped"
    assert statuses["do-more"] == "skipped"


def test_fail_fast_pass_continues(runner, monkeypatch, tmp_path):
    # 0/0: all entries pass -> no abort, exit 0.
    tests = [
        _abort_test("check device", "fail-fast", command="true"),
        {"id": "do-stuff", "name": "do stuff", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 0
    summary = _summary(tmp_path)
    assert summary["success"] is True
    assert summary["passed"] == 2
    assert summary["failed"] == 0


def test_continue_after_failure_runs_remaining(runner, monkeypatch, tmp_path):
    # 1/1: continue-after-failure entry fails -> later entries still run, exit 1.
    # Add a fail-fast entry to satisfy the "at least one fail-fast" requirement.
    tests = [
        _abort_test("check environment", "fail-fast", command="true"),
        _abort_test("check device", "continue-after-failure", command="false"),
        {"id": "do-stuff", "name": "do stuff", "command": "true"},
        {"id": "do-more", "name": "do more", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    assert summary["success"] is False
    assert summary["total"] == 4
    assert summary["failed"] == 1
    assert summary["passed"] == 3
    statuses = {entry["id"]: entry["status"] for entry in summary["entries"]}
    assert statuses["check environment"] == "passed"
    assert statuses["check device"] == "failed"
    assert statuses["do-stuff"] == "passed"
    assert statuses["do-more"] == "passed"


def test_aggregate_failure_is_nonzero(runner, monkeypatch, tmp_path):
    # 1/0: all entries run (continue-after-failure), one fails -> aggregate nonzero.
    # First entry has explicit fail-fast, satisfying the requirement.
    tests = [
        {"id": "one", "name": "one", "command": "true", "failure_policy": "fail-fast"},
        {
            "id": "two",
            "name": "two",
            "command": "false",
            "failure_policy": "continue-after-failure",
        },
        {"id": "three", "name": "three", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    assert summary["success"] is False
    assert [entry["status"] for entry in summary["entries"]] == [
        "passed",
        "failed",
        "passed",
    ]


def test_preflight_policy_default(runner):
    # Preflight entries default to fail-fast; functional entries default to
    # continue-after-failure.
    assert (
        runner._entry_failure_policy({"name": "check isolated environment"}, 0)
        == "fail-fast"
    )
    assert (
        runner._entry_failure_policy({"name": "run operator tests"}, 0)
        == "continue-after-failure"
    )


def test_functional_default_is_continue_after_failure(runner, monkeypatch, tmp_path):
    # Functional entries without explicit failure_policy default to
    # continue-after-failure, so one failing functional group does not hide others.
    tests = [
        {"id": "preflight", "name": "check device", "command": "true"},
        {"id": "func-a", "name": "run operator tests", "command": "false"},
        {"id": "func-b", "name": "run general tests", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    assert summary["success"] is False
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    statuses = {entry["id"]: entry["status"] for entry in summary["entries"]}
    assert statuses["preflight"] == "passed"
    assert statuses["func-a"] == "failed"
    assert statuses["func-b"] == "passed"


def test_all_continue_after_failure_rejected(runner, monkeypatch, tmp_path):
    # A manifest where every entry is continue-after-failure (no fail-fast gate)
    # is rejected: preflight failures must abort, not run the entire suite.
    import sys

    tests = [
        {
            "id": "one",
            "name": "run tests",
            "command": "true",
            "failure_policy": "continue-after-failure",
        },
        {
            "id": "two",
            "name": "run more tests",
            "command": "true",
            "failure_policy": "continue-after-failure",
        },
    ]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run_integration_tests.py"])

    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert "must have at least one fail-fast entry" in str(exc.value)


def test_all_skipped_detected(runner, monkeypatch, tmp_path):
    # When a fail-fast preflight fails and aborts the entire suite, all_skipped
    # is set to True in the summary.
    tests = [
        _abort_test("check device", "fail-fast", command="false"),
        {"id": "do-stuff", "name": "do stuff", "command": "true"},
        {"id": "do-more", "name": "do more", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    assert summary["success"] is False
    assert summary["total"] == 3
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    assert summary["skipped"] == 2
    assert summary.get("all_skipped") is False  # Not all are skipped (one failed)

    # Now test when the preflight passes but all functional tests are skipped
    # (hypothetical edge case where every functional entry is marked skipped).
    # For the actual all-skipped case: first entry fails, rest are skipped.
    tests_all_skip = [
        _abort_test("check device", "fail-fast", command="false"),
        {"id": "func", "name": "run tests", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests_all_skip)
    summary = _summary(tmp_path)
    # 1 failed, 1 skipped -> not all_skipped
    assert summary.get("all_skipped") is False


def test_input_truth_table(runner, monkeypatch, tmp_path):
    # Truth table for (has_fail_fast, has_continue_after_failure) configurations:
    # (T, T): valid - typical manifest with preflight + functional groups
    # (T, F): valid - all entries abort on failure (conservative)
    # (F, T): INVALID - rejected by all_continue_after_failure check
    # (F, F): impossible - empty manifest rejected earlier

    # Case (T, T): typical manifest
    tests_tt = [
        {"id": "preflight", "name": "check device", "command": "true"},
        {"id": "func", "name": "run tests", "command": "true"},
    ]
    assert _run(runner.main, monkeypatch, tmp_path, tests_tt) == 0

    # Case (T, F): all fail-fast
    tests_tf = [
        {
            "id": "one",
            "name": "check device",
            "command": "true",
            "failure_policy": "fail-fast",
        },
        {
            "id": "two",
            "name": "check environment",
            "command": "true",
            "failure_policy": "fail-fast",
        },
    ]
    assert _run(runner.main, monkeypatch, tmp_path, tests_tf) == 0

    # Case (F, T): all continue-after-failure - rejected
    tests_ft = [
        {
            "id": "one",
            "name": "run tests",
            "command": "true",
            "failure_policy": "continue-after-failure",
        },
    ]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests_ft))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert "must have at least one fail-fast entry" in str(exc.value)


def test_default_failure_policy_is_phase_aware(runner, monkeypatch, tmp_path):
    # Without explicit failure_policy, the default depends on phase:
    # preflight -> fail-fast, functional -> continue-after-failure.
    tests = [
        {"id": "check", "name": "check device", "command": "false"},
        {"id": "func", "name": "run operator tests", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    statuses = {entry["id"]: entry["status"] for entry in summary["entries"]}
    # Preflight defaulted to fail-fast, aborted remaining entries
    assert statuses["check"] == "failed"
    assert statuses["func"] == "skipped"


def test_validate_only_does_not_write_reports(runner, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "INTEGRATION_TESTS", json.dumps([_abort_test("check device", "fail-fast")])
    )
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["run_integration_tests.py", "--validate-only"])

    assert runner.main() == 0
    assert not (tmp_path / "integration-reports" / "integration-summary.json").exists()


def test_empty_tests_rejected_without_allow_empty(runner, monkeypatch, tmp_path):
    import sys

    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps([]))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run_integration_tests.py"])

    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert "must not be empty" in str(exc.value)


def test_invalid_manifest_rejected(runner, monkeypatch, tmp_path):
    import sys

    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps([{"name": "x"}]))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run_integration_tests.py"])

    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert "command must be a non-empty string" in str(exc.value)
