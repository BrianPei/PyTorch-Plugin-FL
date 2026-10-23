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

"""Replay the recorded adapter output, so moving data cannot change a decision.

The fixtures under ``tests/unit/fixtures/flagcicd_v1`` are captured before a
structural change and compared afterwards.  A difference here means CI selects,
refuses or blocks something different; regenerate deliberately with
``python3 tests/manual/regenerate_flagcicd_fixtures.py --capture`` and review why.
"""

import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures/flagcicd_v1"

sys.path.insert(0, str(SCRIPTS))
from flagcicd_plan import build_plan  # noqa: E402
from test_tiers import policy, risk_profile  # noqa: E402

sys.path.pop(0)

PLAN_CASES = {
    "no_diff": None,
    "documentation": ["README.md"],
    "native_runtime": ["csrc/runtime.cpp"],
    "platform_config": [".github/configs/nvidia.yml"],
    "mixed": ["README.md", "csrc/runtime.cpp", "tests/unit/test_test_tiers.py"],
}
PLAN_SCALARS = (
    "test_level",
    "risk_level",
    "risk_rule_version",
    "profile",
    "diff_status",
    "requested_level",
    "target_kind",
    "selection",
    "selection_origin",
    "validation_scope",
    "scope",
    "schema_version",
    "manual_test_count",
)
SELECTION_FIELDS = (
    "id",
    "name",
    "category",
    "level",
    "execution",
    "min_passed",
    "command",
    "environment",
    "requirement_evidence",
)


def recorded(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True)


def plan_request():
    return {
        "source_sha": "a" * 40,
        "source_repository": "flagos-ai/Torch-FL",
        "base_sha": "b" * 40,
        "requested_level": "auto",
        "platforms": "all",
        "profile": "auto",
        "task_id": "",
        "parent_run_id": "",
    }


def test_fixture_index_matches_repository():
    index = recorded("manifest")
    assert index["schema_version"] == "flagcicd.fixtures/v1"
    assert index["platforms"] == sorted(
        path.stem for path in (ROOT / ".github/configs").glob("*.yml")
    )
    assert index["plan_cases"] == sorted(PLAN_CASES)
    assert index["files"] == sorted(("plans", "risk_profiles", "selections"))


def test_risk_profiles_are_unchanged():
    recorded_profiles = recorded("risk_profiles")
    assert recorded_profiles == {
        name: risk_profile(changed) for name, changed in PLAN_CASES.items()
    }


def test_plans_are_unchanged():
    contract = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
    captured = recorded("plans")
    for name, changed in PLAN_CASES.items():
        plan = build_plan(plan_request(), contract, ROOT, changed)
        projection = {
            "scalars": {key: plan[key] for key in PLAN_SCALARS},
            "changed_files": plan["changed_files"],
            "matched_rules": plan["matched_rules"],
            "platforms": plan["platforms"],
            "validated_platforms": plan["validated_platforms"],
            "expected_platforms": plan["expected_platforms"],
            "required_platforms": plan["required_platforms"],
            "blocked_cases": plan["blocked_cases"],
            "unsupported_cases": plan["unsupported_cases"],
            "platform_decisions": {
                platform: sorted(
                    (
                        {
                            "id": decision["id"],
                            "status": decision["status"],
                            "reason": decision["reason"],
                        }
                        for decision in decisions
                    ),
                    key=lambda decision: decision["id"],
                )
                for platform, decisions in plan["case_decisions"].items()
            },
            "manual_tests": sorted(plan["manual_tests"]),
        }
        assert projection == {key: captured[name][key] for key in projection}, (
            f"{name}: plan changed"
        )
        # The digest pins the fields the projection leaves out: whole case bodies,
        # manual-test details and the per-case contracts.
        body = {key: value for key, value in plan.items() if key != "config_hash"}
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert digest == captured[name]["digest"], (
            f"{name}: excluded plan fields changed"
        )
        # The contract fingerprint moves whenever the contract text does; record it
        # so an edit is visible here rather than silently absorbed into the digest.
        assert plan["config_hash"] == captured[name]["config_hash"], (
            f"{name}: contract changed; regenerate with --capture"
        )


def test_selections_are_unchanged():
    from compose_integration_manifest import compose

    manifest = yaml.safe_load((ROOT / ".github/integration-manifest.yml").read_text())
    known = sorted(path.stem for path in (ROOT / ".github/configs").glob("*.yml"))
    document = policy()
    captured = recorded("selections")
    assert set(captured) == {
        f"{platform}/{profile}"
        for platform in known
        for profile in ("pr", "pr-risk", "nightly", "weekly", "release")
    }
    for platform in known:
        config = yaml.safe_load((ROOT / f".github/configs/{platform}.yml").read_text())
        for profile in ("pr", "pr-risk", "nightly", "weekly", "release"):
            key = f"{platform}/{profile}"
            try:
                groups = compose(manifest, config, platform, known, profile, document)
            except ValueError as error:
                assert captured[key] == {"status": "refused", "reason": str(error)}, (
                    f"{key}: expected a selection"
                )
                continue
            projected = [
                {field: group[field] for field in SELECTION_FIELDS if field in group}
                for group in groups
            ]
            assert captured[key] == {"status": "selected", "groups": projected}, (
                f"{key}: selection changed"
            )


@pytest.mark.parametrize("name", sorted(PLAN_CASES))
def test_recorded_plan_scalars_are_complete(name):
    """A projection that dropped a scalar would silently stop guarding it."""
    assert set(recorded("plans")[name]["scalars"]) == set(PLAN_SCALARS)
