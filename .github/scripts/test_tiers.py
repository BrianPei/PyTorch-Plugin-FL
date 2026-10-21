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

"""Shared, data-driven test selection; vendor runners need only the stdlib."""

import argparse
import fnmatch
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEVELS = {f"L{number}": number for number in range(5)}
REQUIREMENTS = {"accelerator", "model_directory"}


def policy(root=ROOT):
    document = json.loads((root / ".flagcicd/test-tiers.json").read_text())
    if document.get("schema_version") != "flagcicd.tiers/v1":
        raise ValueError("unsupported test tier schema")
    for name, case in document["cases"].items():
        if (
            case["level"] not in LEVELS
            or type(case["timeout_seconds"]) is not int
            or not 0 < case["timeout_seconds"] <= 7200
        ):
            raise ValueError(f"invalid tier or timeout for {name}")
        profiles = case.get("profiles")
        requires = case.get("requires")
        resources = case.get("resources")
        if (
            not isinstance(profiles, list)
            or not profiles
            or not all(isinstance(p, str) for p in profiles)
            or len(profiles) != len(set(profiles))
            or any(p not in document["profiles"] for p in profiles)
            or not isinstance(requires, list)
            or not all(isinstance(r, str) for r in requires)
            or any(r not in REQUIREMENTS for r in requires)
            or len(requires) != len(set(requires))
            or not isinstance(resources, dict)
            or set(resources) != {"min_accelerators"}
            or type(resources["min_accelerators"]) is not int
            or not 1 <= resources["min_accelerators"] <= 1024
        ):
            raise ValueError(f"invalid profiles, requirements or resources for {name}")
    return document


def exclusion_reason(case_id, case, profile, document):
    if profile not in case["profiles"]:
        return "profile_excluded"
    if LEVELS[case["level"]] > LEVELS[document["profiles"][profile]["max_level"]]:
        return "above_profile_level"
    return ""


def requirement_evidence(case, config):
    """Use the existing environment contract, without inventing device support."""
    environment = config.get("environment_check", {})
    blocked = []
    if (
        "model_directory" in case["requires"]
        and environment.get("model_dir") is not True
    ):
        blocked.append("model directory is not declared by environment_check.model_dir")
    bounds = environment.get("flagos_devices", {})
    minimum = case["resources"]["min_accelerators"]
    exact = bounds.get("exactly")
    if type(exact) is int and exact < minimum:
        blocked.append(
            f"requires {minimum} accelerators; environment declares exactly {exact}"
        )
    return {
        "status": "blocked" if blocked else "requires_runtime_verification",
        "reasons": blocked,
        "source": "environment_check and recorded hardware identity",
    }


def select(groups, ids, profile, platform, document):
    """Classify every group before filtering; unknown groups are errors."""
    selected = []
    definition = document["profiles"][profile]
    if profile == "release" and platform not in document["release"]["model_platforms"]:
        raise ValueError(document["release"]["unavailable_reason"])
    for group, case_id in zip(groups, ids, strict=True):
        case = document["cases"][case_id]
        # The legacy source-contract diagnostic cannot accept an installed wheel.
        if exclusion_reason(case_id, case, profile, document):
            continue
        item = {
            **group,
            "id": case_id,
            "platform": platform,
            "min_passed": 1,
            "selection_reason": f"profile:{profile}",
            **case,
        }
        item["command"] = item["command"].replace(
            "UNIT_TESTS_ALLOW_TEARDOWN_CRASH=1", "UNIT_TESTS_ALLOW_TEARDOWN_CRASH=0"
        )
        # No profile may silently turn a process crash into a passing test.
        item["environment"] = {
            **item.get("environment", {}),
            "INTEGRATION_ALLOW_TEARDOWN_CRASH": "0",
            "UNIT_TESTS_ALLOW_TEARDOWN_CRASH": "0",
        }
        selected.append(item)
    extras = [
        ("l3_installed_contract", "tests/acceptance/test_installed_contract.py"),
        ("l3_lifecycle", "tests/acceptance/test_lifecycle.py"),
        ("l3_performance", "tests/acceptance/test_performance.py"),
    ]
    if definition["max_level"] == "L4":
        extras.append(("l4_endurance", "tests/acceptance/test_endurance.py"))
        if platform in document["release"]["model_platforms"]:
            extras.append(("l4_models", "tests/acceptance/test_models.py"))
    for case_id, path in extras:
        case = document["cases"][case_id]
        if not exclusion_reason(case_id, case, profile, document):
            selected.append(
                {
                    "id": case_id,
                    "name": case_id,
                    "command": f"python -m pytest {path} -v --tb=short",
                    "platform": platform,
                    "min_passed": 1,
                    "selection_reason": f"profile:{profile}",
                    **case,
                    "environment": {
                        "FLAGCICD_PROFILE": profile,
                        "FLAGCICD_STABILITY_SECONDS": str(
                            document["release"][
                                "stability_seconds"
                                if profile == "release"
                                else "weekly_stability_seconds"
                            ]
                        ),
                        "FLAGCICD_MAX_RETAINED_BYTES": str(
                            document["release"]["max_retained_bytes"]
                        ),
                    },
                }
            )
    if not selected or len({g["id"] for g in selected}) != len(selected):
        raise ValueError("empty selection or duplicate case IDs")
    return selected


def risk_profile(changed):
    patterns = (
        "csrc/*",
        "cmake/*",
        "torch_fl/*",
        "setup.py",
        "requirements*",
        "constraints*",
        ".github/version-pins.env",
        ".github/constraints.txt",
        "scripts/tools/torch-fl-preflight",
        ".github/scripts/hooks/*",
        ".github/configs/*",
        ".flagcicd/*",
        "tests/acceptance/*",
    )
    return (
        "pr-risk"
        if changed is None
        or any(
            fnmatch.fnmatchcase(path, pattern)
            for path in changed
            for pattern in patterns
        )
        else "pr"
    )


def check_inventory(root=ROOT):
    """An explicit file inventory makes new, moved and deleted tests visible."""
    inventory = json.loads((root / ".flagcicd/test-inventory.json").read_text())
    actual = {str(p.relative_to(root)) for p in (root / "tests").rglob("test_*.py")}
    actual.update(str(p.relative_to(root)) for p in (root / "tests").rglob("*_test.py"))
    recorded = set(inventory["files"])
    if actual != recorded:
        raise ValueError(
            f"unclassified tests: {sorted(actual - recorded)}; stale tests: {sorted(recorded - actual)}"
        )
    for path, entry in inventory["files"].items():
        if (
            not entry.get("owner")
            or not entry.get("reason")
            or entry["lane"] not in ("L1", "L2", "L3", "L4", "manual")
        ):
            raise ValueError(f"incomplete inventory entry: {path}")
        if entry["lane"] != "manual":
            targets = entry.get("entrypoints", [])
            if not targets or not all((root / target).is_file() for target in targets):
                raise ValueError(f"missing fixed entrypoint: {path}")
            selector = entry.get("selector")
            if not selector or not any(
                selector in (root / target).read_text() for target in targets
            ):
                raise ValueError(f"test lost its fixed selection: {path}")
        elif entry.get("disposition") not in (
            "manual_validation",
            "awaiting_platform_validation",
        ):
            raise ValueError(f"manual test has no explicit disposition: {path}")
    manifest_path = root / ".github/integration-manifest.yml"
    if manifest_path.exists():
        import yaml
        from compose_integration_manifest import validate_manifest, known_platforms

        manifest = yaml.safe_load(manifest_path.read_text())
        _, case_ids, _ = validate_manifest(
            manifest, known_platforms(root / ".github/configs")
        )
        missing = set(case_ids) - set(policy(root)["cases"])
        if missing:
            raise ValueError(
                f"manifest groups have no tier classification: {sorted(missing)}"
            )
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--changed", type=Path)
    parser.add_argument("--event-profile", action="store_true")
    args = parser.parse_args()
    if args.event_profile:
        from ci.generate_matrix import _changed_files

        event = os.environ.get("EVENT_NAME")
        if event == "schedule":
            selected = "nightly"
        elif event == "push":
            selected = "pr"
        else:
            selected = risk_profile(
                _changed_files(
                    os.environ.get("BASE_REF", "main"),
                    os.environ.get("BASE_SHA", ""),
                    os.environ.get("HEAD_SHA", ""),
                )
            )
        print("profile=" + selected)
    if args.inventory:
        inventory = check_inventory()
        print(
            f"Classified {len(inventory['files'])} test files (manual exclusions remain explicit)"
        )
        for path, entry in inventory["files"].items():
            if entry["lane"] == "manual":
                print(f"Manual gap: {path}: {entry['reason']} Owner: {entry['owner']}")
    if args.changed:
        print("profile=" + risk_profile(json.loads(args.changed.read_text())))


if __name__ == "__main__":
    main()
