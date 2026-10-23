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

"""Capture the composed selection and plan output that the adapter refactor must preserve.

Run with ``--capture`` to rewrite the fixtures and with ``--check`` to verify that the
current code still reproduces them.  The fixtures are the contract for a structural
change: they let a refactor move data around without altering what CI selects.

This script is manual tooling.  It is deliberately not under ``scripts/tools``, which
``setup.py`` packages as installed entry points.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
FIXTURES = Path(__file__).resolve().parents[1] / "unit/fixtures/flagcicd_v1"
PROFILES = ("pr", "pr-risk", "nightly", "weekly", "release")
PLAN_CASES = {
    "no_diff": None,
    "documentation": ["README.md"],
    "native_runtime": ["csrc/runtime.cpp"],
    "platform_config": [".github/configs/nvidia.yml"],
    "mixed": ["README.md", "csrc/runtime.cpp", "tests/unit/test_test_tiers.py"],
}


def load(name):
    # ``flagcicd_plan`` imports ``ci.generate_matrix``, which only resolves with the
    # script directory itself on the path.
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def platforms():
    return sorted(path.stem for path in (ROOT / ".github/configs").glob("*.yml"))


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


def compose_selections(composer, manifest, config, platform, known, profile, document):
    """Copy the composed command and the derived decision, not the copied policy.

    ``profiles``, ``requires``, ``resources`` and ``timeout_seconds`` are bytes from
    ``test-tiers.json`` that ``select`` pastes into every group; comparing them here
    would only repeat the tier document fifty times.
    """
    return [
        {field: group[field] for field in SELECTION_FIELDS if field in group}
        for group in composer(manifest, config, platform, known, profile, document)
    ]


def selections():
    compose_integration_manifest = load("compose_integration_manifest")
    test_tiers = load("test_tiers")
    manifest = yaml.safe_load((ROOT / ".github/integration-manifest.yml").read_text())
    known = platforms()
    document = test_tiers.policy()
    captured = {}
    for platform in known:
        config = yaml.safe_load((ROOT / f".github/configs/{platform}.yml").read_text())
        for profile in PROFILES:
            key = f"{platform}/{profile}"
            try:
                captured[key] = {
                    "status": "selected",
                    "groups": compose_selections(
                        compose_integration_manifest.compose,
                        manifest,
                        config,
                        platform,
                        known,
                        profile,
                        document,
                    ),
                }
            except ValueError as error:
                captured[key] = {"status": "refused", "reason": str(error)}
    return captured


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


def project_plan(plan):
    """Keep the decisions a structural change can move, not the repeated case bodies.

    ``case_decisions``, ``cases`` and ``manual_tests`` restate the whole policy for
    every platform; storing them five times would add about a megabyte of duplicated
    data.  Each decision is projected to the id, status and reason that a selection
    change would surface, and the digest pins the full document byte for byte.

    ``config_hash`` is the fingerprint of the whole ``project.yaml`` file, so it
    changes whenever the contract is edited even if nothing it drives changes.  It is
    recorded as its own field and left out of the digest; otherwise the structural
    refactor this fixture exists to verify could never reproduce its own baseline.
    """
    body = {key: value for key, value in plan.items() if key != "config_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {
        "config_hash": plan["config_hash"],
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
        "digest": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def plans():
    flagcicd_plan = load("flagcicd_plan")
    contract = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
    captured = {}
    for name, changed in PLAN_CASES.items():
        request = {
            "source_sha": "a" * 40,
            "source_repository": "flagos-ai/Torch-FL",
            "base_sha": "b" * 40,
            "requested_level": "auto",
            "platforms": "all",
            "profile": "auto",
            "task_id": "",
            "parent_run_id": "",
        }
        plan = flagcicd_plan.build_plan(request, contract, ROOT, changed)
        captured[name] = project_plan(plan)
    return captured


def risk_profiles():
    test_tiers = load("test_tiers")
    return {
        name: test_tiers.risk_profile(changed) for name, changed in PLAN_CASES.items()
    }


def collect():
    return {
        "risk_profiles": risk_profiles(),
        "plans": plans(),
        "selections": selections(),
    }


def manifest(captured):
    return {
        "schema_version": "flagcicd.fixtures/v1",
        "platforms": platforms(),
        "profiles": list(PROFILES),
        "plan_cases": sorted(PLAN_CASES),
        "files": sorted(captured),
    }


def write(captured):
    # Compact output: these files are regenerated whole, never read as a diff, and
    # indentation is a third of their size.
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "manifest.json").write_text(
        json.dumps(manifest(captured), sort_keys=True) + "\n"
    )
    for name, document in captured.items():
        target = FIXTURES / f"{name}.json"
        target.write_text(json.dumps(document, sort_keys=True) + "\n")
        print(f"wrote {target.relative_to(ROOT)}")


def check(captured):
    failed = []
    index = FIXTURES / "manifest.json"
    if not index.is_file():
        failed.append("manifest.json: missing fixture index")
    elif json.loads(index.read_text()) != manifest(captured):
        failed.append("manifest.json: differs from the recorded index")
    for name, document in captured.items():
        target = FIXTURES / f"{name}.json"
        if not target.is_file():
            failed.append(f"{name}: missing fixture")
            continue
        recorded = json.loads(target.read_text())
        if recorded != json.loads(json.dumps(document, sort_keys=True)):
            failed.append(f"{name}: differs from the recorded fixture")
    if failed:
        print("\n".join(failed), file=sys.stderr)
        return 1
    print("fixtures match the current selection and plan output")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="rewrite the fixtures")
    parser.add_argument("--check", action="store_true", help="verify the fixtures")
    args = parser.parse_args()
    if not args.capture and not args.check:
        parser.error("pass --capture or --check")
    captured = collect()
    if args.capture:
        write(captured)
    if args.check:
        return check(captured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
