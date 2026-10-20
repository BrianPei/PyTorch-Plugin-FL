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

"""Build an auditable test plan from one request, one diff and one contract.

The plan answers four questions in writing, so a reviewer can check the
answers instead of inferring them from a green run:

* which level is being proven, and why that level
  (``level_sources`` records each input that raised it);
* which platforms run, and why each one is in or out of the matrix
  (``selection_reason`` per platform);
* which groups are scheduled, and why a group is not
  (``not_selected`` carries a reason per group);
* how the result may be compared later (``case_set_hash`` and
  ``config_hash`` bind the run to the exact case set and inputs).

Every refusal here is deliberate. An unsupported level, a level whose
acceptance record has no evidence, a requested platform that has no config,
and a profile that requires a level the repository cannot prove all raise
rather than producing an empty matrix -- an empty matrix reports success
without running anything, which is the failure mode this file exists to make
impossible.
"""

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
)
from compose_integration_manifest import ManifestError, select


LEVELS = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


class PlanError(Exception):
    """A request the contract cannot honour, stated in terms a user can act on."""


def contract_root():
    """The trusted checkout root: this script lives at <root>/.github/scripts."""
    return Path(__file__).resolve().parents[2]


def load_contract(root):
    """Read the capability contract and the acceptance record it references."""
    root = Path(root)
    contract = yaml.safe_load((root / ".flagcicd/project.yaml").read_text())
    acceptance = yaml.safe_load((root / ".flagcicd/acceptance.yaml").read_text())
    if contract.get("schema_version") != "flagcicd.project/v1":
        raise PlanError("unsupported capability contract schema")
    if not isinstance(contract.get("profiles"), dict) or not contract["profiles"]:
        raise PlanError("the contract must declare at least one profile")
    if not isinstance(contract.get("levels"), dict):
        raise PlanError("the contract must declare its levels")
    return contract, acceptance


def level_support(contract, acceptance, level):
    """Whether a level may run, and the reason when it may not.

    ``supported`` is a claim the contract makes about code that exists.
    ``pending_acceptance`` is a claim about code plus hardware evidence, so it
    needs a record: status ``accepted``, at least one evidence entry, and the
    control commit that produced it. Anything else is refused with the reason
    the record states, not with a generic "unsupported".
    """
    declared = contract["levels"].get(level)
    if not isinstance(declared, dict):
        raise PlanError(f"level {level} is not declared in the capability contract")
    support = declared.get("support")
    if support == "supported":
        return True, "declared supported"
    if support == "unsupported":
        return False, f"level {level} is declared unsupported"
    if support == "pending_acceptance":
        record = (acceptance.get("levels") or {}).get(level) or {}
        if record.get("status") != "accepted":
            return False, (
                f"level {level} is pending hardware acceptance "
                f"(status: {record.get('status', 'missing')})"
            )
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            return False, f"level {level} is accepted without evidence"
        if (
            not isinstance(record.get("accepted_sha"), str)
            or not record["accepted_sha"]
        ):
            return False, f"level {level} is accepted without a control commit"
        return True, f"accepted by {record.get('owner') or 'an unnamed owner'}"
    return False, f"level {level} states an unknown support value {support!r}"


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


def deepest_supported(contract, acceptance):
    """The deepest level this repository can currently prove.

    A profile states the depth it *wants*; whether that depth may run is a
    separate fact, and it changes while a level waits for hardware acceptance.
    Reading it here, once, keeps the profile minimum and the acceptance record
    from disagreeing silently.
    """
    deepest = None
    for level in sorted(LEVELS, key=LEVELS.__getitem__):
        if level_support(contract, acceptance, level)[0]:
            deepest = level
    return deepest


def effective_level(request, contract, inferred, acceptance):
    """The level is the maximum of request, diff and profile minimum.

    A profile that runs the whole platform set with a cheap case set still
    states a minimum depth, so a docs-only nightly cannot quietly stop
    proving the installed wheel.

    The one adjustment is a profile minimum deeper than the deepest level with
    acceptance evidence: ``weekly`` asks for L3, L3 is ``pending_acceptance``,
    and refusing the whole schedule would mean the L2 work that *is* runnable
    never happens either. The profile's ask is capped at the deepest provable
    level and the shortfall is recorded in ``level_deferred`` so the plan says
    "L2, wanted L3, L3 not accepted" rather than passing off L2 as the weekly
    budget. An explicit ``requested_level`` or a diff that demands the deeper
    level is not capped: those are honoured or refused, never rewritten.
    """
    profile = request["profile"]
    declared = contract["profiles"][profile]
    requested = request["requested_level"]
    sources = {
        "requested": "L0" if requested == "auto" else requested,
        "diff": inferred,
        "profile": declared["minimum_level"],
    }
    level = max(sources.values(), key=LEVELS.__getitem__)
    deferred = None
    if level == sources["profile"] and level not in (sources["requested"], sources["diff"]):
        ceiling = deepest_supported(contract, acceptance)
        if ceiling is not None and LEVELS[level] > LEVELS[ceiling]:
            ok, reason = level_support(contract, acceptance, level)
            deferred = {
                "wanted": level,
                "applied": ceiling,
                "reason": reason if not ok else "level is not provable",
            }
            level = max(ceiling, sources["requested"], sources["diff"], key=LEVELS.__getitem__)
    raised_by = sorted(
        name for name, value in sources.items() if value == level and level != "L0"
    )
    return level, sources, raised_by, deferred


def select_platforms(request, known, configs, changed):
    """Resolve the matrix, keeping the request's explicit list authoritative.

    ``platforms=all`` keeps its historical meaning: every platform for
    ``selection=all`` (nightly/weekly), the change-aware subset for
    ``selection=affected`` (pull requests). A dispatch that names platforms
    overrides both. Labels only ever add, never subtract: a ``platform:<name>``
    label unions with the affected set and with each other, and the widened
    scope is recorded in ``selection_reason`` so a reader can see the plan was
    not the file diff's own answer.
    """
    labels = request.get("labels") or []
    requested = request["platforms"]
    if not isinstance(requested, str):
        # The resolver normalizes a dispatch array into the comma-separated
        # form, so a non-string here is a hand-built request. Refuse it here
        # rather than letting it reach the platform-name parser as a type error.
        raise PlanError("platforms must be a string in the normalized request")
    selected = None

    if request["selection"] == "affected":
        affected = _platforms_from_files(changed, known, configs)
        if request["platforms"] == "all":
            selected = list(affected or [])
            selected = _apply_platform_labels(selected, labels, known)
    if request["platforms"] != "all":
        selected = _platforms_from_dispatch(requested, known)

    if selected is None:
        selected = known if request["selection"] == "all" else []

    selected = [platform for platform in known if platform in set(selected)]
    widened = LEVELS[request["level"]] >= LEVELS["L2"] and not selected
    if widened:
        # L2 and above install a wheel on hardware. An empty matrix would
        # report success without doing so, so the plan widens instead.
        selected = list(known)

    reason = {}
    for platform in known:
        if platform not in selected:
            continue
        if request["platforms"] != "all":
            reason[platform] = "the request names this platform"
        elif widened:
            reason[platform] = (
                "widened: the level builds a wheel and no platform was affected"
            )
        elif request["selection"] == "all":
            reason[platform] = "selection=all runs every configured platform"
        elif _label_platforms(labels, known) is not None:
            reason[platform] = "platform labels add this platform"
        elif changed is None:
            reason[platform] = "the diff is unavailable, so every platform runs"
        else:
            reason[platform] = "the diff touches a path this platform watches"
    return selected, reason


def _label_platforms(labels, known):
    prefix = "platform:"
    routing = [
        label[len(prefix) :].strip() for label in labels if label.startswith(prefix)
    ]
    if not routing:
        return None
    if "all" in routing or any(platform not in known for platform in routing):
        return list(known)
    return [platform for platform in known if platform in set(routing)]


def _apply_platform_labels(selected, labels, known):
    """Union labels into the file-derived set; a label can only add scope."""
    label_platforms = _label_platforms(labels, known)
    if label_platforms is None:
        return selected
    return [
        platform
        for platform in known
        if platform in set(selected) | set(label_platforms)
    ]


def _groups_for(platform, manifest, config, known, contract, level, profile):
    """Select the platform's groups and describe every exclusion."""
    cases = contract["profiles"][profile]["cases"]
    return select(
        manifest,
        config,
        platform,
        known,
        level=level,
        profile=profile,
        cases=cases,
        level_order=LEVELS,
    )


def build_plan(request, contract, acceptance, source, changed, level=None):
    known = _load_platforms(source / ".github/configs")
    if not known:
        raise PlanError("no platform configurations found")

    inferred, risk, matched, diff_status = infer_level(changed, contract)
    level, level_sources, raised_by, deferred = effective_level(
        request, contract, inferred, acceptance
    )
    ok, support_reason = level_support(contract, acceptance, level)
    if not ok:
        raise PlanError(support_reason)

    profile = request["profile"]
    if profile not in contract["profiles"]:
        raise PlanError(f"unknown profile {profile!r}")

    selected, selection_reason = select_platforms(
        {**request, "level": level}, known, source / ".github/configs", changed
    )

    manifest_path = source / ".github/integration-manifest.yml"
    manifest = yaml.safe_load(manifest_path.read_text())
    fingerprint = hashlib.sha256(manifest_path.read_bytes())
    groups, not_selected = {}, {}
    case_set = hashlib.sha256()
    for platform in selected:
        config_path = source / ".github/configs" / f"{platform}.yml"
        config = yaml.safe_load(config_path.read_text())
        chosen, skipped = _groups_for(
            platform, manifest, config, known, contract, level, profile
        )
        groups[platform] = chosen
        not_selected[platform] = skipped
        case_set.update(platform.encode())
        case_set.update(json.dumps([group["id"] for group in chosen]).encode())
        fingerprint.update(platform.encode())
        fingerprint.update(config_path.read_bytes())
    fingerprint.update(json.dumps(contract, sort_keys=True).encode())
    fingerprint.update(case_set.digest())

    return {
        "schema_version": "flagcicd.plan/v1",
        **request,
        "test_level": level,
        "scope": contract["levels"][level]["scope"],
        "level_sources": level_sources,
        "level_raised_by": raised_by,
        "level_deferred": deferred,
        "level_support": support_reason,
        "risk_level": risk,
        "risk_rule_version": contract["risk_rules"]["version"],
        "diff_status": diff_status,
        "changed_files": changed,
        "matched_rules": matched,
        "profile_cases": contract["profiles"][profile]["cases"],
        "config_hash": fingerprint.hexdigest(),
        "case_set_hash": case_set.hexdigest(),
        "platforms": selected,
        "validated_platforms": selected or known,
        "expected_platforms": selected if LEVELS[level] >= LEVELS["L2"] else [],
        "selection_reason": selection_reason,
        "groups": groups,
        "not_selected": not_selected,
    }


def plan_matrix(plan):
    """The hardware matrix, one entry per expected platform.

    The entry carries the platform's selected group ids so the runner does not
    re-decide what to run. Both entries consume this output, which is why it
    is computed here rather than in each workflow: a second computation is a
    second answer, and the two would eventually disagree.
    """
    return [
        {"platform": platform, "groups": json.dumps(plan["groups"][platform])}
        for platform in plan["expected_platforms"]
    ]


def report_context(plan):
    """Keep potentially large diffs in artifacts, outside process environments.

    Groups are summarized by their hash here and carried in full by the plan
    artifact; the platform report re-reads them from the artifact directory, so
    the environment stays far below the 16 KiB dispatch limit.
    """
    fields = (
        "schema_version",
        "source_repository",
        "source_sha",
        "base_sha",
        "control_sha",
        "profile",
        "selection",
        "test_level",
        "level_deferred",
        "scope",
        "risk_level",
        "risk_rule_version",
        "config_hash",
        "case_set_hash",
        "platforms",
        "expected_platforms",
        "run_id",
        "run_attempt",
        "task_id",
        "parent_run_id",
    )
    encoded = json.dumps({key: plan[key] for key in fields}, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16384:
        raise PlanError("report context exceeds the 16 KiB dispatch limit")
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=None,
        help="trusted checkout holding .flagcicd (default: this script's repository)",
    )
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    contract, acceptance = load_contract(args.contract_dir or contract_root())

    if request.get("base_sha"):
        previous = Path.cwd()
        try:
            os.chdir(args.source)
            changed = _changed_files("main", request["base_sha"], request["source_sha"])
        finally:
            os.chdir(previous)
    else:
        # No comparison commit is not "nothing changed": it is an unanswerable
        # question, and infer_level reads it at the contract's conservative
        # level for unknown changes.
        changed = None

    try:
        plan = build_plan(request, contract, acceptance, args.source, changed)
    except (PlanError, ManifestError) as error:
        # A manifest defect is as much a refusal as a request the contract
        # cannot honour: both mean no plan, and both should read as one
        # actionable line rather than a traceback through two modules.
        print(f"::error::{error}")
        raise SystemExit(1) from error
    plan.update(
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
    )
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("context=" + report_context(plan) + "\n")
        output.write(f"test_level={plan['test_level']}\n")
        output.write(f"profile={plan['profile']}\n")
        output.write("platforms=" + json.dumps(plan["expected_platforms"]) + "\n")
        output.write(
            "validated_platforms=" + json.dumps(plan["validated_platforms"]) + "\n"
        )
        output.write(
            "matrix=" + json.dumps(plan_matrix(plan), separators=(",", ":")) + "\n"
        )
    print(
        f"Plan: {plan['profile']} {plan['test_level']} ({plan['risk_level']}), "
        f"platforms={plan['expected_platforms']}, "
        f"{sum(len(v) for v in plan['groups'].values())} group(s)"
    )


if __name__ == "__main__":
    main()
