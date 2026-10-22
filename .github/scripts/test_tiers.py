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
REQUIREMENTS = {
    "accelerator",
    "model_directory",
    "metrics_directory",
    "primary_device",
}
EXECUTION_MODES = {"parallel", "exclusive"}
# The adapter files this module reads.  Kept here because the module is stdlib only
# and cannot parse the YAML contract that documents the same paths; a contract test
# keeps the two in step.
ADAPTER = {
    "tiers": ".flagcicd/test-tiers.json",
    "environment": ".flagcicd/environment-contract.json",
    "acceptance": ".flagcicd/release-acceptance.json",
    "inventory": ".flagcicd/test-inventory.json",
}


def _adapter(root, name):
    return Path(root) / ADAPTER[name]


def environment_contract(root=ROOT):
    """Read the resource rules a repository answers with its own platform config."""
    document = json.loads(_adapter(root, "environment").read_text())
    if document.get("schema_version") != "flagcicd.environment/v1":
        raise ValueError("unsupported environment contract schema")
    if not isinstance(document.get("source"), str) or not document["source"].strip():
        raise ValueError("environment contract has no evidence source")
    rules = document.get("requirement_rules")
    if not isinstance(rules, dict) or any(
        name not in REQUIREMENTS
        or not isinstance(rule, dict)
        or not isinstance(rule.get("path"), str)
        or rule.get("condition") != "is_true"
        or not isinstance(rule.get("reason"), str)
        for name, rule in rules.items()
    ):
        raise ValueError("invalid environment requirement rules")
    bounds = document.get("device_bounds")
    if (
        not isinstance(bounds, dict)
        or not isinstance(bounds.get("path"), str)
        or not isinstance(bounds.get("exact_key"), str)
        or not isinstance(bounds.get("reason"), str)
    ):
        raise ValueError("invalid environment device bounds")
    substitutions = document.get("command_substitutions")
    if not isinstance(substitutions, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in substitutions.items()
    ):
        raise ValueError("invalid environment command substitutions")
    forced = document.get("forced_environment")
    if not isinstance(forced, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in forced.items()
    ):
        raise ValueError("invalid forced environment")
    return document


def acceptance_policy(root=ROOT):
    """Read the acceptance tests this repository adds above the manifest."""
    document = json.loads(_adapter(root, "acceptance").read_text())
    if document.get("schema_version") != "flagcicd.acceptance/v1":
        raise ValueError("unsupported acceptance schema")
    extras = document.get("extras")
    if not isinstance(extras, list) or not extras:
        raise ValueError("acceptance contract has no extras")
    for entry in extras:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not isinstance(entry.get("path"), str)
            or entry.get("when_level") not in (*LEVELS, None)
            or not isinstance(entry.get("path"), str)
            or not entry["path"].strip()
        ):
            raise ValueError(f"invalid acceptance extra: {entry}")
    return document


def risk_patterns(root=ROOT):
    """Read the changed paths that make a diff risk bearing."""
    document = json.loads(_adapter(root, "tiers").read_text())
    patterns = document.get("risk_profile", {}).get("patterns")
    if (
        not isinstance(patterns, list)
        or not patterns
        or not all(isinstance(pattern, str) and pattern for pattern in patterns)
    ):
        raise ValueError("invalid risk profile patterns")
    return patterns


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
        execution = case.get("execution")
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
            or "min_accelerators" not in resources
            or set(resources) - {"min_accelerators", "shared"}
            or type(resources["min_accelerators"]) is not int
            or not 1 <= resources["min_accelerators"] <= 1024
            or ("shared" in resources and type(resources["shared"]) is not bool)
        ):
            raise ValueError(f"invalid profiles, requirements or resources for {name}")
        if execution is not None and execution not in EXECUTION_MODES:
            raise ValueError(f"invalid execution mode for {name}")
        unsupported = case.get("unsupported_platforms", {})
        if not isinstance(unsupported, dict) or any(
            not isinstance(platform, str)
            or not platform.strip()
            or not isinstance(reason, str)
            or not reason.strip()
            for platform, reason in unsupported.items()
        ):
            raise ValueError(f"invalid unsupported platforms for {name}")
        if LEVELS[case["level"]] in (2, 3) and execution is None:
            raise ValueError(
                f"L2/L3 case {name} must declare an execution mode explicitly"
            )
        if execution == "parallel" and resources.get("shared") is not True:
            raise ValueError(f"parallel case {name} must declare resources.shared=true")
    return document


def exclusion_reason(case_id, case, profile, document, platform=None):
    if profile not in case["profiles"]:
        return "profile_excluded"
    if LEVELS[case["level"]] > LEVELS[document["profiles"][profile]["max_level"]]:
        return "above_profile_level"
    if platform in case.get("unsupported_platforms", {}):
        return "platform_unsupported"
    return ""


def requirement_evidence(case, config, contract=None):
    """Use the repository's environment contract, without inventing device support."""
    contract = environment_contract() if contract is None else contract
    environment = config.get("environment_check", {})
    blocked = []
    for requirement in case["requires"]:
        rule = contract["requirement_rules"].get(requirement)
        if rule is not None and environment.get(rule["path"]) is not True:
            blocked.append(rule["reason"])
    bounds = environment.get(contract["device_bounds"]["path"], {})
    minimum = case["resources"]["min_accelerators"]
    exact = bounds.get(contract["device_bounds"]["exact_key"])
    if type(exact) is int and exact < minimum:
        blocked.append(
            contract["device_bounds"]["reason"].format(minimum=minimum, exact=exact)
        )
    return {
        "status": "blocked" if blocked else "requires_runtime_verification",
        "reasons": blocked,
        "source": contract["source"],
    }


def _lookup(document, path):
    value = document
    for name in path.split("."):
        value = value[name]
    return value


def select(groups, ids, profile, platform, document, acceptance=None, contract=None):
    """Classify every group before filtering; unknown groups are errors."""
    contract = environment_contract() if contract is None else contract
    acceptance = acceptance_policy() if acceptance is None else acceptance
    selected = []
    definition = document["profiles"][profile]
    if profile == "release" and platform not in document["release"]["model_platforms"]:
        raise ValueError(document["release"]["unavailable_reason"])
    for group, case_id in zip(groups, ids, strict=True):
        case = document["cases"][case_id]
        # The legacy source-contract diagnostic cannot accept an installed wheel.
        if exclusion_reason(case_id, case, profile, document, platform):
            continue
        item = {
            **group,
            "id": case_id,
            "platform": platform,
            "min_passed": 1,
            "selection_reason": f"profile:{profile}",
            **case,
        }
        # Unclassified cases are isolated by default.  Shared execution must
        # be an explicit policy decision validated above.
        item["execution"] = case.get("execution", "exclusive")
        for original, replacement in contract["command_substitutions"].items():
            item["command"] = item["command"].replace(original, replacement)
        # No profile may silently turn a process crash into a passing test.
        item["environment"] = {
            **item.get("environment", {}),
            **contract["forced_environment"],
        }
        selected.append(item)
    for extra in acceptance["extras"]:
        case_id = extra["id"]
        case = document["cases"][case_id]
        if (
            extra.get("when_level")
            and LEVELS[definition["max_level"]] < LEVELS[extra["when_level"]]
        ):
            continue
        if "platforms_source" in extra and platform not in _lookup(
            document, extra["platforms_source"]
        ):
            continue
        if not exclusion_reason(case_id, case, profile, document, platform):
            selected.append(
                {
                    "id": case_id,
                    "name": case_id,
                    "command": f"python -m pytest {extra['path']} -v --tb=short",
                    "platform": platform,
                    "min_passed": 1,
                    "selection_reason": f"profile:{profile}",
                    **case,
                    "execution": case.get("execution", "exclusive"),
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
                    "stream_output": True,
                }
            )
    if not selected or len({g["id"] for g in selected}) != len(selected):
        raise ValueError("empty selection or duplicate case IDs")
    return selected


def risk_profile(changed, root=ROOT):
    patterns = risk_patterns(root)
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
