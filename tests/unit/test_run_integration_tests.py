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
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "run_integration_tests.py"
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
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({"FOO": "bar"}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
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
    tests = [
        _abort_test("check device", "continue-after-failure", command="false"),
        {"id": "do-stuff", "name": "do stuff", "command": "true"},
        {"id": "do-more", "name": "do more", "command": "true"},
    ]
    code = _run(runner.main, monkeypatch, tmp_path, tests)

    assert code == 1
    summary = _summary(tmp_path)
    assert summary["success"] is False
    assert summary["total"] == 3
    assert summary["failed"] == 1
    assert summary["passed"] == 2
    statuses = {entry["id"]: entry["status"] for entry in summary["entries"]}
    assert statuses["check device"] == "failed"
    assert statuses["do-stuff"] == "passed"
    assert statuses["do-more"] == "passed"


def test_aggregate_failure_is_nonzero(runner, monkeypatch, tmp_path):
    # 1/0: all entries run (continue-after-failure), one fails -> aggregate nonzero.
    tests = [
        {"id": "one", "name": "one", "command": "true"},
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
    # A name containing a preflight marker defaults to fail-fast; anything else
    # defaults to the script-level fail-fast constant.
    assert (
        runner._entry_failure_policy(
            {"name": "check isolated environment"}, 0
        )
        == "fail-fast"
    )
    assert (
        runner._entry_failure_policy({"name": "run operator tests"}, 0)
        == runner.DEFAULT_FAILURE_POLICY
    )


def test_validate_only_does_not_write_reports(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps([_abort_test("check device", "fail-fast")]))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["run_integration_tests.py", "--validate-only"])

    assert runner.main() == 0
    assert not (tmp_path / "integration-reports" / "integration-summary.json").exists()


def test_empty_tests_rejected_without_allow_empty(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps([]))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert "must not be empty" in str(exc.value)


def test_invalid_manifest_rejected(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps([{"name": "x"}]))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert "command must be a non-empty string" in str(exc.value)
