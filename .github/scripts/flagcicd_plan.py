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

"""Build an auditable test plan from the source diff and one capability contract."""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path

import yaml

from ci.generate_matrix import (
    _changed_files,
    _load_platforms,
    _platforms_from_dispatch,
    _platforms_from_files,
    _platforms_from_labels,
)
from compose_integration_manifest import compose
from test_tiers import exclusion_reason, policy, risk_profile


LEVELS = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def infer_level(changed, contract):
    policy = contract["risk_rules"]
    if changed is None:
        return policy["no_diff_level"], policy["unknown_risk"], [], "unavailable"
    level, risk, matched = "L0", "R0", []
    for path in changed:
        path = path[2:] if path.startswith("./") else path
        rules = [
            rule
            for rule in policy["rules"]
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule["paths"])
        ]
        if not rules:
            rules = [
                {
                    "id": "unknown",
                    "minimum_level": policy["unknown_level"],
                    "risk": policy["unknown_risk"],
                }
            ]
        for rule in rules:
            level = max((level, rule["minimum_level"]), key=LEVELS.__getitem__)
            risk = max(risk, rule["risk"])
            matched.append({"path": path, "rule": rule["id"]})
    return level, risk, matched, "available"


def build_plan(request, contract, source, changed):
    requested = request["requested_level"]
    if requested not in ("auto", *LEVELS):
        raise ValueError("unsupported test level")
    configs = source / ".github/configs"
    known = _load_platforms(configs)
    if not known:
        raise ValueError("no platform configurations found")
    selection = request["platforms"]
    if isinstance(selection, list):
        if not selection or not all(
            isinstance(p, str) and p in known for p in selection
        ):
            raise ValueError("platforms must be a non-empty array of platform names")
        selection = ",".join(selection)
    if not isinstance(selection, str) or not selection.strip():
        raise ValueError(
            "platforms must be all, a comma-separated list, or a JSON array"
        )
    selected = _platforms_from_dispatch(selection, known)
    candidates = selected
    mode = request.get("selection", "affected" if selection.strip() == "all" else "all")
    if mode not in ("affected", "all"):
        raise ValueError("selection must be affected or all")
    labels = request.get("pr_labels", [])
    if not isinstance(labels, list) or not all(isinstance(lb, str) for lb in labels):
        raise ValueError("PR labels must be an array of strings")
    routed = _platforms_from_labels(labels, known)
    if routed is not None:
        selected = routed
    elif mode == "affected" and changed is not None:
        affected = _platforms_from_files(changed, known, configs)
        selected = [platform for platform in selected if platform in affected]
    inferred, risk, matched, diff_status = infer_level(changed, contract)
    level = max(
        (inferred, "L0" if requested == "auto" else requested), key=LEVELS.__getitem__
    )
    if routed is not None:
        level = max((level, "L2"), key=LEVELS.__getitem__)
    profile = request.get("profile", "pr")
    tiers = policy(source)
    if profile == "auto":
        profile = risk_profile(changed)
        if level == "L3":
            profile = "pr-risk"
    elif profile not in tiers["profiles"] or profile == "release":
        raise ValueError("unsupported profile; release acceptance needs a candidate")
    if request.get("profile") not in (None, "auto") or profile == "pr-risk":
        level = max(
            (level, tiers["profiles"][profile]["level"]), key=LEVELS.__getitem__
        )
    if level == "L3" and profile == "pr":
        profile = "pr-risk"
    if contract["levels"][level]["support"] != "supported":
        raise ValueError(f"test level {level} is not supported")
    # An explicit hardware request must not become an empty, passing matrix.
    if level in ("L2", "L3") and not selected:
        selected = candidates
    if profile in ("nightly", "weekly") and selection.strip() == "all":
        selected = known
    required_platforms = _platforms_from_files(changed, known, configs)
    if profile in ("nightly", "weekly"):
        required_platforms = known
    diagnostic = level in ("L2", "L3") and bool(set(required_platforms) - set(selected))
    validation_scope = "diagnostic" if diagnostic else "full"
    platform_decisions = [
        {
            "platform": platform,
            "status": "selected"
            if level in ("L2", "L3") and platform in selected
            else "excluded",
            "reason": (
                "hardware_not_required"
                if level in ("L0", "L1")
                else "platform_label"
                if routed is not None
                else "scheduled_profile"
                if profile in ("nightly", "weekly")
                else "candidate_platform_set"
                if platform not in candidates
                else "all_candidates"
                if mode == "all"
                else "affected_or_hardware_floor"
                if platform in selected
                else "not_affected"
            ),
        }
        for platform in known
    ]
    manifest_path = source / ".github/integration-manifest.yml"
    manifest = yaml.safe_load(manifest_path.read_text())
    fingerprint = hashlib.sha256(manifest_path.read_bytes())
    selections = {}
    case_decisions = {}
    blocked_cases = []
    case_contracts = {}
    fingerprint.update(json.dumps(tiers, sort_keys=True).encode())
    for name in ("test-inventory.json", "performance-baselines.json"):
        fingerprint.update((source / ".flagcicd" / name).read_bytes())
    for platform in selected or known:
        path = configs / f"{platform}.yml"
        config = yaml.safe_load(path.read_text())
        groups = compose(manifest, config, platform, known, profile, tiers)
        for group in groups:
            requirement = {
                "min_accelerators": group["resources"]["min_accelerators"],
                "min_passed": group["min_passed"]
                if "pytest" in group["command"]
                or "run_unit_tests.py" in group["command"]
                else 0,
            }
            previous = case_contracts.setdefault(group["id"], requirement)
            if previous != requirement:
                raise ValueError(f"inconsistent execution contract for {group['id']}")
        selections[platform] = [
            {
                key: group[key]
                for key in (
                    "id",
                    "level",
                    "category",
                    "requires",
                    "resources",
                    "timeout_seconds",
                )
            }
            for group in groups
        ]
        group_map = {group["id"]: group for group in groups}
        case_decisions[platform] = []
        applicable = {
            entry["id"]
            for entry in manifest["steps"]
            if platform in entry.get("platforms", known)
        }
        for case_id, case in tiers["cases"].items():
            group = group_map.get(case_id)
            if group:
                reason = group.get("blocked_reason", "selected_by_profile")
                state = "blocked" if group.get("blocked_reason") else "selected"
                evidence = group["requirement_evidence"]
                if level in ("L0", "L1"):
                    state, reason = "excluded", "hardware_not_required"
                if (
                    state == "blocked"
                    and platform in selected
                    and level in ("L2", "L3")
                ):
                    blocked_cases.append(
                        {"platform": platform, "id": case_id, "reason": reason}
                    )
            else:
                reason = exclusion_reason(case_id, case, profile, tiers) or (
                    "platform_not_applicable"
                    if case_id not in applicable
                    else "not_selected"
                )
                state, evidence = "excluded", {}
            case_decisions[platform].append(
                {
                    "id": case_id,
                    "status": state,
                    "reason": reason,
                    "requirements": evidence,
                }
            )
        fingerprint.update(platform.encode())
        fingerprint.update(path.read_bytes())
    fingerprint.update(json.dumps(contract, sort_keys=True).encode())
    manual_tests = {
        path: entry
        for path, entry in json.loads(
            (source / ".flagcicd/test-inventory.json").read_text()
        )["files"].items()
        if entry["lane"] == "manual"
    }
    expected_cases = {
        p: [g["id"] for g in sorted(groups, key=lambda g: g["level"])]
        for p, groups in selections.items()
    }
    # Reruns may only reuse reports for the same execution policy and control code.
    fingerprint.update(
        json.dumps(
            {
                "profile": profile,
                "test_level": level,
                "selection": mode,
                "validation_scope": validation_scope,
                "required_platforms": required_platforms,
                "expected_cases": expected_cases,
                **{
                    key: request.get(key, "")
                    for key in ("control_sha", "target_kind", "base_sha", "diff_sha")
                },
            },
            sort_keys=True,
        ).encode()
    )
    return {
        "schema_version": "flagcicd.plan/v1",
        **request,
        **{
            key: request.get(key, "")
            for key in ("control_sha", "target_kind", "diff_sha")
        },
        "selection": mode,
        "selection_origin": request.get("selection_origin", "legacy_direct_request"),
        "profile": profile,
        "cases": selections,
        "platform_decisions": platform_decisions,
        "case_decisions": case_decisions,
        "case_contracts": case_contracts,
        "blocked_cases": blocked_cases,
        "required_platforms": required_platforms,
        "validation_scope": validation_scope,
        "manual_tests": manual_tests,
        "manual_test_count": len(manual_tests),
        "expected_cases": expected_cases,
        "test_level": level,
        "scope": contract["levels"][level]["scope"],
        "risk_level": risk,
        "risk_rule_version": contract["risk_rules"]["version"],
        "diff_status": diff_status,
        "changed_files": changed,
        "matched_rules": matched,
        "config_hash": fingerprint.hexdigest(),
        "platforms": selected,
        "validated_platforms": selected or known,
        "expected_platforms": selected if level in ("L2", "L3") else [],
    }


def report_context(plan):
    """Keep potentially large diffs in artifacts, outside process environments."""
    fields = (
        "schema_version",
        "source_repository",
        "source_sha",
        "base_sha",
        "test_level",
        "scope",
        "risk_level",
        "risk_rule_version",
        "config_hash",
        "platforms",
        "expected_platforms",
        "run_id",
        "run_attempt",
        "task_id",
        "parent_run_id",
    )
    encoded = json.dumps(
        {
            **{key: plan[key] for key in fields},
            "profile": plan.get("profile", "pr"),
            **{
                key: plan.get(key, "")
                for key in ("control_sha", "target_kind", "diff_sha", "selection")
            },
            "expected_cases": plan.get("expected_cases", {}),
            **{
                key: plan.get(key, default)
                for key, default in (
                    ("selection_origin", ""),
                    ("validation_scope", "full"),
                    ("required_platforms", []),
                    ("blocked_cases", []),
                    ("platform_decisions", []),
                    ("case_contracts", {}),
                )
            },
            "manual_test_count": len(plan.get("manual_tests", {})),
            "plan_artifact": f"flagcicd-plan-{plan['run_attempt']}",
        },
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > 16384:
        raise ValueError("report context exceeds the 16 KiB dispatch limit")
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    contract = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".flagcicd/project.yaml").read_text()
    )
    previous = Path.cwd()
    try:
        os.chdir(args.source)
        changed = _changed_files(
            "main", request["base_sha"], request.get("diff_sha", request["source_sha"])
        )
    finally:
        os.chdir(previous)
    plan = build_plan(request, contract, args.source, changed)
    plan.update(
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
    )
    plan["plan_artifact"] = f"flagcicd-plan-{plan['run_attempt']}"
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("context=" + report_context(plan) + "\n")
        output.write(f"profile={plan['profile']}\n")
        output.write(f"test_level={plan['test_level']}\n")
        output.write("platforms=" + json.dumps(plan["expected_platforms"]) + "\n")
        output.write(
            "validated_platforms=" + json.dumps(plan["validated_platforms"]) + "\n"
        )
    print(
        f"Plan: {plan['test_level']} ({plan['risk_level']}), platforms={plan['expected_platforms']}"
    )


if __name__ == "__main__":
    main()
