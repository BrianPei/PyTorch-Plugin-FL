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

"""Exercise tier selection, lost test evidence and immutable release candidates."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
from candidate_evidence import finish, sha256, verify, verify_installation  # noqa: E402
from compose_integration_manifest import compose  # noqa: E402
from merge_pytest_reports import merge_reports  # noqa: E402
from flagcicd_plan import build_plan, report_context  # noqa: E402
from test_tiers import check_inventory, policy, risk_profile  # noqa: E402

sys.path.pop(0)


def test_repository_inventory_is_complete():
    check_inventory()


@pytest.mark.parametrize(
    "case_id,filename",
    [
        ("cuda_flaggems_mul", "test_flaggems_mul_route.py"),
        ("cuda_dtype_cast_performance", "test_dtype_cast_performance.py"),
    ],
)
def test_cuda_profiling_runs_only_in_cuda_l3_profiles(case_id, filename):
    from manifest_composer import composer, manifest, platform_config, platform_names

    document = policy()
    known = platform_names()
    for platform in known:
        for profile in document["profiles"]:
            if (
                profile == "release"
                and platform not in document["release"]["model_platforms"]
            ):
                continue
            groups = composer.compose(
                manifest(),
                platform_config(platform),
                platform,
                known,
                profile,
                document,
            )
            selected = [g for g in groups if g["id"] == case_id]
            assert bool(selected) == (platform == "cuda" and profile != "pr"), (
                platform,
                profile,
            )
            if selected:
                assert selected[0]["level"] == "L3"
                assert selected[0]["execution"] == "exclusive"
                assert selected[0]["min_passed"] == 1
                assert f"tests/integration/ops/{filename}" in selected[0]["command"]


def test_unregistered_file_is_rejected(tmp_path):
    (tmp_path / ".flagcicd").mkdir()
    (tmp_path / ".flagcicd/test-inventory.json").write_text('{"files": {}}')
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_forgotten.py").touch()
    with pytest.raises(ValueError, match="unclassified"):
        check_inventory(tmp_path)


@pytest.mark.parametrize(
    "platform", sorted(p.stem for p in (ROOT / ".github/configs").glob("*.yml"))
)
def test_profiles_classify_every_manifest_case(platform):
    manifest = yaml.safe_load((ROOT / ".github/integration-manifest.yml").read_text())
    config = yaml.safe_load((ROOT / f".github/configs/{platform}.yml").read_text())
    known = [p.stem for p in (ROOT / ".github/configs").glob("*.yml")]
    pr = compose(manifest, config, platform, known, "pr")
    nightly = compose(manifest, config, platform, known, "nightly")
    assert all(g["level"] == "L2" for g in pr)
    assert {g["id"] for g in pr} <= {g["id"] for g in nightly}
    assert {"l3_installed_contract", "l3_lifecycle"} <= {g["id"] for g in nightly}
    assert ("multi_device_contracts" in {g["id"] for g in nightly}) == (
        platform != "mps"
    )
    assert "l4_endurance" in {g["id"] for g in nightly}
    if platform in policy()["release"]["model_platforms"]:
        assert "l4_models" in {g["id"] for g in nightly}
    assert [g["level"] for g in nightly] == sorted(g["level"] for g in nightly)
    assert all("UNIT_TESTS_ALLOW_TEARDOWN_CRASH=1" not in g["command"] for g in nightly)


def test_release_refuses_unconfigured_model_acceptance():
    from test_tiers import select

    with pytest.raises(ValueError, match="Model mounts"):
        select([], [], "release", "ascend", policy())


def test_candidate_acceptance_excludes_source_only_diagnostics():
    from test_tiers import select

    selected = select(
        [{"name": "Source sweep", "command": 'cd "$GITHUB_WORKSPACE" && pytest'}],
        ["unit_tests_per_file"],
        "release",
        "cuda",
        policy(),
    )
    assert "unit_tests_per_file" not in {group["id"] for group in selected}


@pytest.mark.parametrize(
    "changed,expected",
    [
        (None, "pr-risk"),
        (["README.md"], "pr"),
        (["csrc/runtime.cpp"], "pr-risk"),
        (["setup.py"], "pr-risk"),
    ],
)
def test_runtime_changes_get_bounded_l3(changed, expected):
    assert risk_profile(changed) == expected


def evidence():
    receipt = {
        "source_sha": "a" * 40,
        "source_repository": "flagos-ai/Torch-FL",
        "platform": "cuda",
    }
    groups = [
        {"name": c, "category": c, "status": "passed"}
        for c in ("compatibility", "resource", "stability", "model", "performance")
    ]
    report = {
        "exitcode": 0,
        "summary": {"passed": 4},
        "flagcicd": {
            "complete": True,
            "source_ref": receipt["source_sha"],
            "source_repository": receipt["source_repository"],
            "platform": "cuda",
            "execution": {
                "complete": True,
                "expected": [g["name"] for g in groups],
                "groups": groups,
            },
        },
    }
    hardware = {"accelerator_count": 1, "sdk_version": "verified-sdk"}
    return receipt, report, hardware, policy()["release"]


def test_complete_evidence_accepts_candidate():
    assert finish(*evidence())["status"] == "accepted"


@pytest.mark.parametrize(
    "defect",
    [
        "skip",
        "missing_group",
        "missing_model",
        "failed",
        "wrong_source",
        "unknown_sdk",
        "missing_device",
    ],
)
def test_partial_evidence_cannot_become_release_verified(defect):
    receipt, report, hardware, requirements = evidence()
    if defect == "skip":
        report["summary"]["skipped"] = 1
    elif defect == "missing_group":
        report["flagcicd"]["execution"]["groups"].pop()
    elif defect == "missing_model":
        next(
            g
            for g in report["flagcicd"]["execution"]["groups"]
            if g["category"] == "model"
        )["category"] = "functional"
    elif defect == "failed":
        report["flagcicd"]["execution"]["groups"][0]["status"] = "failed"
    elif defect == "wrong_source":
        report["flagcicd"]["source_ref"] = "b" * 40
    elif defect == "unknown_sdk":
        hardware["sdk_version"] = "unknown"
    else:
        hardware["accelerator_count"] = 0
    with pytest.raises(ValueError):
        finish(receipt, report, hardware, requirements)


def test_publisher_refuses_different_bytes_and_wrong_run(tmp_path):
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"accepted bytes")
    receipt = {
        "source_sha": "a" * 40,
        "platform": "cuda",
        "run_id": "42",
        "wheel_sha256": sha256(wheel),
        "wheel": wheel.name,
        "status": "accepted",
        "profile": "release",
    }
    verify(receipt, wheel, "a" * 40, "cuda", "42")
    with pytest.raises(ValueError):
        verify(receipt, wheel, "a" * 40, "cuda", "43")
    wheel.write_bytes(b"rebuilt bytes")
    with pytest.raises(ValueError):
        verify(receipt, wheel, "a" * 40, "cuda", "42")


def test_installed_package_must_match_candidate_archive(tmp_path, monkeypatch):
    import candidate_evidence

    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torch_fl/__init__.py", b"candidate = True\n")
    installed = tmp_path / "installed"
    (installed / "torch_fl").mkdir(parents=True)
    package = installed / "torch_fl/__init__.py"
    package.write_bytes(b"candidate = True\n")

    class Distribution:
        def locate_file(self, name):
            return installed / name

    monkeypatch.setattr(
        candidate_evidence.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    verify_installation(wheel)
    package.write_bytes(b"stale = True\n")
    with pytest.raises(ValueError, match="differs from candidate"):
        verify_installation(wheel)


def test_nightly_all_is_not_narrowed_by_documentation_diff():
    contract = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
    request = {
        "source_sha": "a" * 40,
        "source_repository": "flagos-ai/Torch-FL",
        "base_sha": "b" * 40,
        "requested_level": "auto",
        "platforms": "all",
        "profile": "nightly",
        "task_id": "",
        "parent_run_id": "",
    }
    plan = build_plan(request, contract, ROOT, ["README.md"])
    assert plan["test_level"] == "L3"
    assert plan["expected_platforms"] == sorted(
        p.stem for p in (ROOT / ".github/configs").glob("*.yml")
    )
    plan.update(run_id="42", run_attempt="1")
    assert len(report_context(plan).encode()) <= 16384
    request["profile"] = "auto"
    assert build_plan(request, contract, ROOT, ["README.md"])["test_level"] == "L0"
    assert (
        build_plan(request, contract, ROOT, ["csrc/runtime.cpp"])["test_level"] == "L3"
    )


def test_real_pytest_collection_detects_lost_case(tmp_path):
    if importlib.util.find_spec("pytest_jsonreport") is None:
        pytest.fail("pytest-json-report is required to validate the real collector")
    (tmp_path / "test_example.py").write_text("def test_example(): pass\n")
    reports = tmp_path / "reports"
    reports.mkdir()
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(SCRIPTS / "pytest_reporting"),
        "PYTEST_PLUGINS": "pytest_jsonreport.plugin,flagcicd_report",
        "FLAGCICD_REPORT_DIR": str(reports),
        "FLAGCICD_TEST_GROUP": "example",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_example.py", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert merge_reports(reports, "cuda", "success", "repo", "sha")["flagcicd"][
        "complete"
    ]
    marker = next(reports.glob("*.pending"))
    data = json.loads(marker.read_text())
    data["selected_nodeids"].append("test_example.py::test_missing")
    marker.write_text(json.dumps(data))
    assert not merge_reports(reports, "cuda", "success", "repo", "sha")["flagcicd"][
        "complete"
    ]


def test_missing_selected_inventory_without_runner_ledger_is_incomplete(tmp_path):
    """Direct acceptance pytest sessions still require collection evidence."""
    (tmp_path / "acceptance.pending").write_text(
        json.dumps({"group": "l3_performance", "args": ["test_performance.py"]})
    )
    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {
                "summary": {"passed": 1},
                "tests": [
                    {"nodeid": "test_performance.py::test_fixed_workload_performance"}
                ],
                "exitcode": 0,
                "duration": 0.1,
            }
        )
    )
    merged = merge_reports(tmp_path, "cuda", "success", "repo", "sha")
    assert merged["flagcicd"]["complete"] is False
    assert merged["flagcicd"]["incomplete_sessions"] == ["acceptance.json"]


def test_timeout_is_not_success_and_unstarted_groups_remain_visible(tmp_path):
    groups = [
        {
            "name": "hang",
            "command": f'{sys.executable} -c "import time; time.sleep(20)"',
            "level": "L3",
            "timeout_seconds": 0.1,
        },
        {"name": "unreached", "command": "true"},
    ]
    env = {
        **os.environ,
        "INTEGRATION_TESTS": json.dumps(groups),
        "FLAGCICD_REPORT_DIR": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_integration_tests.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 124
    ledger = json.loads((tmp_path / "groups.json").read_text())
    assert ledger["groups"][0]["status"] == "timeout"
    assert not ledger["complete"]
    assert len(ledger["expected"]) == 2 and len(ledger["groups"]) == 1


def test_release_graph_gates_all_lanes_and_reuses_candidates():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    assert jobs["build"]["with"]["publish"] is False
    assert {"lint", "preflight", "agnostic", "codegen", "wheel-checks"} <= set(
        jobs["build"]["needs"]
    )
    assert (
        jobs["acceptance"]["with"]["candidate_artifact"]
        == "wheel-${{ matrix.platform }}"
    )
    assert "acceptance" in jobs["publish"]["needs"]
    assert (
        jobs["build"]["strategy"]["matrix"]
        == jobs["acceptance"]["strategy"]["matrix"]
        == jobs["publish"]["strategy"]["matrix"]
    )


def test_all_profiled_commands_disable_legacy_crash_tolerance():
    doc = copy.deepcopy(policy())
    assert doc["profiles"]["weekly"]["max_level"] == "L4"
    assert doc["profiles"]["nightly"]["max_level"] == "L4"
    assert doc["release"]["stability_seconds"] >= 3600


def test_execution_modes_default_shared_groups_and_isolate_sensitive_groups():
    from test_tiers import select

    document = policy()
    selected = select(
        [
            {"name": "Factory", "command": "true"},
            {"name": "Environment", "command": "true"},
        ],
        ["factory_ops", "environment"],
        "nightly",
        "cuda",
        document,
    )
    modes = {group["id"]: group["execution"] for group in selected}
    assert modes["factory_ops"] == "parallel"
    assert modes["l4_endurance"] == "exclusive"
    # Preflights gate the suites after them, and device tracers are
    # process-wide: neither may overlap another group.
    assert modes["environment"] == "exclusive"
    for case_id in (
        "musa_runtime",
        "iluvatar_runtime",
        "profiler_contract",
        "profiler_parity",
        "cuda_flaggems_mul",
        "cuda_dtype_cast_performance",
        "multi_device_contracts",
        "native_comm_bridge",
        "torch_compile",
        "inference",
        "training",
        "l3_installed_contract",
        "l3_lifecycle",
        "l3_performance",
        "l4_models",
    ):
        assert document["cases"][case_id]["execution"] == "exclusive", case_id
    # Shared groups must fit on one accelerator; the runner rejects others.
    for case_id, case in document["cases"].items():
        if case.get("execution", "parallel") == "parallel" and case["level"] in (
            "L2",
            "L3",
        ):
            assert case["resources"]["min_accelerators"] == 1, case_id
            assert case["resources"].get("shared") is True, case_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("profiles", []),
        ("profiles", ["unknown"]),
        ("profiles", [["pr"]]),
        ("requires", ["invented_capability"]),
        ("requires", [["accelerator"]]),
        ("resources", {"min_accelerators": True}),
        ("resources", {"min_accelerators": 0}),
        ("resources", {"min_accelerators": 1, "unverified": 42}),
        ("execution", "overlap"),
    ],
)
def test_invalid_case_requirements_fail_before_dispatch(tmp_path, field, value):
    doc = policy()
    doc["cases"]["environment"][field] = value
    (tmp_path / ".flagcicd").mkdir()
    (tmp_path / ".flagcicd/test-tiers.json").write_text(json.dumps(doc))
    message = (
        "invalid execution mode"
        if field == "execution"
        else "profiles, requirements or resources"
    )
    with pytest.raises(ValueError, match=message):
        policy(tmp_path)


def test_l2_l3_case_requires_explicit_execution_mode(tmp_path):
    doc = policy()
    doc["cases"]["factory_ops"].pop("execution")
    (tmp_path / ".flagcicd").mkdir()
    (tmp_path / ".flagcicd/test-tiers.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="must declare an execution mode"):
        policy(tmp_path)


def test_parallel_case_requires_declared_shared_resource(tmp_path):
    doc = policy()
    doc["cases"]["factory_ops"]["resources"].pop("shared")
    (tmp_path / ".flagcicd").mkdir()
    (tmp_path / ".flagcicd/test-tiers.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="resources.shared=true"):
        policy(tmp_path)


@pytest.mark.parametrize(
    "platform", sorted(p.stem for p in (ROOT / ".github/configs").glob("*.yml"))
)
def test_case_metadata_does_not_rewrite_existing_l2_commands(platform):
    manifest = yaml.safe_load((ROOT / ".github/integration-manifest.yml").read_text())
    config = yaml.safe_load((ROOT / f".github/configs/{platform}.yml").read_text())
    known = [p.stem for p in (ROOT / ".github/configs").glob("*.yml")]
    original = compose(manifest, config, platform, known)
    applicable_ids = [
        g["id"] for g in manifest["steps"] if platform in g.get("platforms", known)
    ]
    expected = [
        g
        for g, case in zip(original, applicable_ids, strict=True)
        if policy()["cases"][case]["level"] == "L2"
    ]
    selected = compose(manifest, config, platform, known, "pr")
    assert [(g["name"], g["command"]) for g in selected] == [
        (
            g["name"],
            g["command"].replace(
                "UNIT_TESTS_ALLOW_TEARDOWN_CRASH=1", "UNIT_TESTS_ALLOW_TEARDOWN_CRASH=0"
            ),
        )
        for g in expected
    ]
    assert all(
        g["requires"] == ["accelerator"]
        and g["resources"]["min_accelerators"] == 1
        and (g["execution"] != "parallel" or g["resources"].get("shared") is True)
        for g in selected
    )
    assert all(
        g["requirement_evidence"]["status"] == "requires_runtime_verification"
        for g in selected
    )


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ({"exactly": 1}, "blocked"),
        ({"exactly": 2}, "requires_runtime_verification"),
        ({"at_least": 1}, "requires_runtime_verification"),
        ({}, "requires_runtime_verification"),
    ],
)
def test_declared_lower_bound_is_not_mistaken_for_measured_resources(bounds, expected):
    from test_tiers import requirement_evidence

    case = policy()["cases"]["multi_device_contracts"]
    result = requirement_evidence(
        case, {"environment_check": {"flagos_devices": bounds}}
    )
    assert result["status"] == expected
    assert bool(result["reasons"]) is (expected == "blocked")


@pytest.mark.parametrize("value", [[], {"mps": ""}, {"mps": None}, {"": "reason"}])
def test_unsupported_case_needs_a_platform_and_reason(tmp_path, value):
    document = policy()
    document["cases"]["multi_device_contracts"]["unsupported_platforms"] = value
    (tmp_path / ".flagcicd").mkdir()
    (tmp_path / ".flagcicd/test-tiers.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="invalid unsupported platforms"):
        policy(tmp_path)


@pytest.mark.parametrize(
    "filename", ["test_installed_contract.py", "test_lifecycle.py"]
)
def test_acceptance_initializes_backend_before_torch(monkeypatch, tmp_path, filename):
    import builtins
    import types

    spec = importlib.util.spec_from_file_location(
        "acceptance_under_test", ROOT / "tests/acceptance" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = types.SimpleNamespace(
        __file__=str(tmp_path / "wheel/torch_fl/__init__.py")
    )
    package = Path(backend.__file__).parent
    package.mkdir(parents=True)
    (package / "compatibility.json").write_text('{"build": {}}')
    original_import = builtins.__import__
    initialized = False

    def guarded_import(name, *args, **kwargs):
        nonlocal initialized
        if name == "torch_fl":
            initialized = True
            return backend
        if name == "torch":
            assert initialized, "vendor backend must select its libtorch first"
            raise RuntimeError("torch reached after backend initialization")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    target = (
        module.test_installed_candidate_and_dependencies
        if filename == "test_installed_contract.py"
        else module.cycle
    )
    with pytest.raises(
        RuntimeError, match="torch reached after backend initialization"
    ):
        if filename == "test_installed_contract.py":
            target(lambda *args: None)
        else:
            target()


@pytest.mark.parametrize(
    ("platform", "profile", "whole_environment_check"),
    [("cuda", "pr-risk", False), ("cuda", "release", True), ("metax", "pr-risk", True)],
)
def test_installed_acceptance_keeps_checks_within_the_declared_scope(
    monkeypatch, tmp_path, platform, profile, whole_environment_check
):
    import sys
    import types

    spec = importlib.util.spec_from_file_location(
        "installed_acceptance", ROOT / "tests/acceptance/test_installed_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "installed/torch_fl"
    package.mkdir(parents=True)
    (package / "compatibility.json").write_text(json.dumps({"platform": platform}))
    monkeypatch.setitem(
        sys.modules,
        "torch_fl",
        types.SimpleNamespace(__file__=str(package / "__init__.py")),
    )
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(__version__="2.10.0")
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "2.10.0")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path / "checkout"))
    monkeypatch.setenv("FLAGCICD_PROFILE", profile)
    commands = []
    monkeypatch.setattr(
        module.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd)
    )
    properties = {}
    module.test_installed_candidate_and_dependencies(properties.__setitem__)
    assert properties == (
        {"flagcx_communication": "not_validated"} if platform == "cuda" else {}
    )
    if whole_environment_check:
        assert commands.pop(0) == [sys.executable, "-m", "pip", "check"]
    preflight = [
        "torch-fl-preflight",
        "--installed",
        "--check-installed",
        "--check-build-env",
        "--strict-tested",
    ]
    if profile == "release":
        preflight += ["--release", "--require-sdk"]
    assert commands == [preflight]


def test_performance_uses_published_device_properties_and_writes_evidence(
    monkeypatch, tmp_path
):
    import builtins
    import types

    spec = importlib.util.spec_from_file_location(
        "performance_under_test", ROOT / "tests/acceptance/test_performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tensor:
        def __init__(self, value, size=512 * 512):
            self.value, self.size = value, size

        def __mul__(self, other):
            return Tensor(self.value * other.value, self.size)

        def __add__(self, other):
            return Tensor(self.value + other.value, self.size)

        def sum(self):
            return Tensor(self.value * self.size, 1)

        def cpu(self):
            return self

        def item(self):
            return self.value

    runtime = types.SimpleNamespace(
        synchronize=lambda: None,
        get_device_properties=lambda index: types.SimpleNamespace(name="test-device"),
    )
    backend = types.SimpleNamespace(
        __file__=str(tmp_path / "wheel/torch_fl/__init__.py"), flagos=runtime
    )
    package = Path(backend.__file__).parent
    package.mkdir(parents=True)
    (package / "compatibility.json").write_text('{"build": {"sdk_version": "test"}}')
    workspace = tmp_path / "workspace"
    (workspace / ".flagcicd").mkdir(parents=True)
    (workspace / ".flagcicd/performance-baselines.json").write_text(
        '{"environments": {}}'
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("FLAGCICD_METRICS_DIR", str(metrics))
    monkeypatch.setenv("FLAGCICD_PROFILE", "pr-risk")
    original_import = builtins.__import__
    initialized = False

    def guarded_import(name, *args, **kwargs):
        nonlocal initialized
        if name == "torch_fl":
            initialized = True
            return backend
        if name == "torch":
            assert initialized, "vendor backend must select its libtorch first"
            return types.SimpleNamespace(ones=lambda *args, **kwargs: Tensor(1))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module.test_fixed_workload_performance()
    evidence = json.loads((metrics / "performance.json").read_text())
    assert evidence["environment"]["device"] == "test-device"
    assert len(evidence["samples_seconds"]) == 20
    assert evidence["baseline_status"] == "uncalibrated"
    monkeypatch.setenv("FLAGCICD_PROFILE", "release")
    with pytest.raises(AssertionError, match="approved performance baseline"):
        module.test_fixed_workload_performance()
