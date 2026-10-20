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

"""Expand the shared integration manifest for one platform.

The manifest (.github/integration-manifest.yml) is the one ordered
list of groups the integration pipeline runs. A platform config is data: it
has no step list of its own, and cannot add, drop or reword a step. What a
platform *can* do is state the values a step needs -- its device
expectations, the environment its runner sets -- and this script folds those
into the step so `run_integration_tests.py` receives the same
`integration_tests` JSON it always has.

That shape is the point. Eight per-platform step lists made a step that only
one platform ran look like a step every platform ran; here the deviation is a
`platforms:` key in one file, and how much is still per-platform is a number
the loader prints on every run.

``compose()`` is what the loader job prints: every group a platform runs, which
is the set the pipeline must still execute in one pass. ``select()`` is what
the planner calls: the subset that a level and a profile pay for, plus a written
reason for every group it leaves out. Both read the same entries and produce
the same step shape, so a group cannot be described one way in the plan and run
another way on the runner.

Input is JSON, not YAML: this runs in the loader job on ubuntu-latest, where
`yq` is already the YAML reader, and the vendor images are not required to
carry a Python YAML module. The parsing side of the seam is `yq -o=json`; this
side is the standard library alone, so it can also be imported by the unit
tests that hold the manifest's invariants.
"""

import argparse
import json
import sys
from pathlib import Path


# The one substitution the composer performs. Anything else that looks like a
# variable belongs to the shell on the hardware runner.
DISPLAY_NAME_TOKEN = "{{display_name}}"

# Verification depths, in order. A group declares the shallowest level that can
# prove it; it runs at that level and every deeper one.
LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}

# The case classes a profile can buy. A group that declares none belongs to
# `baseline`, which every profile includes -- a group is only skipped when the
# manifest says so in a way a reviewer can read.
CASE_CLASSES = ("baseline", "regression", "compatibility", "lifecycle", "soak")

# Integration groups install a wheel on hardware, so L2 is the default: a group
# that forgets to declare its level runs rather than quietly disappearing from
# the deepest profile that does not happen to name it.
DEFAULT_LEVEL = "L2"
DEFAULT_CASES = ("baseline",)

_ENTRY_KEYS = {
    "id",
    "name",
    "command",
    "platforms",
    "environment",
    "environment_from_config",
    "level",
    "profiles",
    "cases",
    "requires",
    "timeout_minutes",
    "resources",
}


class ManifestError(Exception):
    """A manifest or platform config that cannot be composed."""


def known_platforms(config_dir) -> list[str]:
    """Every platform that has a config, by basename.

    Read from the directory rather than from a list in the manifest: a
    platform is a file, and a manifest that names a platform no file defines
    would otherwise drop that step for every platform at once and look fine.

    Nothing else may live in that directory. It is also the directory ci.yml's
    Discover platforms step and generate_matrix.py enumerate, so a second kind
    of `.yml` in there becomes a platform and gets scheduled hardware: the
    manifest sits at `.github/integration-manifest.yml` for that reason, and
    this function reads the directory without a name to skip.
    """
    directory = Path(config_dir)
    if not directory.is_dir():
        raise ManifestError(f"config directory not found: {directory}")
    return sorted(path.stem for path in directory.glob("*.yml"))


def _require_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _entry_platforms(entry, known):
    platforms = entry.get("platforms")
    if platforms is None:
        return list(known)
    if not isinstance(platforms, list) or not platforms:
        raise ManifestError(
            f"step {entry.get('id')!r}: platforms must be a non-empty list"
        )
    for platform in platforms:
        if platform not in known:
            raise ManifestError(
                f"step {entry.get('id')!r}: platforms names {platform!r}, which is "
                f"not a platform config; known platforms are {', '.join(known)}"
            )
    if len(set(platforms)) != len(platforms):
        raise ManifestError(f"step {entry.get('id')!r}: platforms repeats a platform")
    return list(platforms)


def _string_list(value, field, allowed=None):
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{field} must be a non-empty list")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ManifestError(f"{field} must contain non-empty strings")
        if allowed is not None and item not in allowed:
            raise ManifestError(
                f"{field} names {item!r}, which is not one of {', '.join(allowed)}"
            )
    if len(set(value)) != len(value):
        raise ManifestError(f"{field} repeats a value")
    return list(value)


def entry_metadata(entry, field):
    """The execution metadata one entry declares, with its defaults filled in.

    Every key is optional because the manifest predates levels and profiles;
    the defaults are the conservative ones. A group that says nothing runs at
    the shallowest level that proves an installed wheel, in every profile, in
    the baseline case set -- so omitting the keys keeps a group running, and
    skipping a group always takes a key someone can grep for.
    """
    level = entry.get("level", DEFAULT_LEVEL)
    if level not in LEVEL_ORDER:
        raise ManifestError(
            f"{field}.level must be one of {', '.join(LEVEL_ORDER)}; got {level!r}"
        )
    profiles = entry.get("profiles")
    if profiles is not None:
        profiles = _string_list(profiles, f"{field}.profiles")
    cases = _string_list(
        entry.get("cases", list(DEFAULT_CASES)), f"{field}.cases", CASE_CLASSES
    )
    requires = entry.get("requires")
    if requires is not None:
        requires = _string_list(requires, f"{field}.requires")
    timeout = entry.get("timeout_minutes")
    if timeout is not None:
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
            raise ManifestError(f"{field}.timeout_minutes must be a positive integer")
    resources = entry.get("resources")
    if resources is not None:
        if not isinstance(resources, dict) or not resources:
            raise ManifestError(f"{field}.resources must be a non-empty mapping")
        for key, value in resources.items():
            _require_string(key, f"{field}.resources key")
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise ManifestError(f"{field}.resources[{key}] must be a scalar")
    return {
        "level": level,
        "profiles": profiles,
        "cases": cases,
        "requires": requires or [],
        "timeout_minutes": timeout,
        "resources": dict(resources or {}),
    }


def validate_manifest(manifest, known):
    """Check the manifest's shape and return its entries.

    The invariants that matter: every entry is complete, a step's variants sit
    together and never overlap (a platform that got the same step twice would
    run it twice, silently), every entry names a platform that exists, and the
    variants of one id agree on what that id means -- its level, its profiles,
    its case classes and what it requires. A group that means one thing on CUDA
    and another on Ascend makes the plan's case set hash a claim about nothing.
    """
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest must be a mapping with a steps list")
    steps = manifest.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ManifestError("the manifest must define a non-empty steps list")

    seen_order: list[str] = []
    assigned: dict[str, set[str]] = {}
    metadata: dict[str, dict] = {}
    for index, entry in enumerate(steps):
        if not isinstance(entry, dict):
            raise ManifestError(f"steps[{index}] must be a mapping")
        unknown = set(entry) - _ENTRY_KEYS
        if unknown:
            raise ManifestError(
                f"steps[{index}] has unknown key(s): {', '.join(sorted(unknown))}"
            )
        step_id = _require_string(entry.get("id"), f"steps[{index}].id")
        _require_string(entry.get("name"), f"steps[{index}].name")
        _require_string(entry.get("command"), f"steps[{index}].command")
        platforms = _entry_platforms(entry, known)
        declared = entry_metadata(entry, f"steps[{index}]")
        for requirement in declared["requires"]:
            if requirement == step_id:
                raise ManifestError(
                    f"step {step_id!r} requires itself, so it could never run"
                )

        if entry.get("environment") is not None:
            _environment(entry, f"steps[{index}].environment")
        environment_from = entry.get("environment_from_config")
        if environment_from is not None:
            if not isinstance(environment_from, dict) or not environment_from:
                raise ManifestError(
                    f"steps[{index}].environment_from_config must be a non-empty "
                    "mapping of platform config key to environment variable name"
                )
            for key, variable in environment_from.items():
                _require_string(key, f"steps[{index}].environment_from_config key")
                _require_string(
                    variable, f"steps[{index}].environment_from_config[{key}]"
                )

        if step_id in assigned:
            previous = seen_order[-1] if seen_order else None
            if previous != step_id:
                raise ManifestError(
                    f"step {step_id!r} is not contiguous: all of a step's variants "
                    "belong together, so collapsing them into one is a local edit"
                )
            overlap = assigned[step_id] & set(platforms)
            if overlap:
                raise ManifestError(
                    f"step {step_id!r} lists {', '.join(sorted(overlap))} in more "
                    "than one variant, so those platforms would run it twice"
                )
            if metadata[step_id] != declared:
                raise ManifestError(
                    f"step {step_id!r} declares different level, profiles, cases or "
                    "requires on different platforms; a group has one meaning"
                )
            assigned[step_id].update(platforms)
        else:
            assigned[step_id] = set(platforms)
            metadata[step_id] = declared
            seen_order.append(step_id)

    # A requirement must name an earlier id: execution follows this file's
    # order, so a dependency declared later would never have run when the
    # dependent group starts.
    position = {step_id: index for index, step_id in enumerate(seen_order)}
    for step_id in seen_order:
        for requirement in metadata[step_id]["requires"]:
            if requirement not in position:
                raise ManifestError(
                    f"step {step_id!r} requires {requirement!r}, which is not a "
                    "manifest id"
                )
            if position[requirement] >= position[step_id]:
                raise ManifestError(
                    f"step {step_id!r} requires {requirement!r}, which runs later "
                    "or at the same position; requirements must run first"
                )

    return steps, seen_order, assigned


def _environment(entry, field):
    environment = entry.get("environment")
    if environment is None:
        # A step that needs none is the common case, and validate_manifest
        # accepts the key's absence; composing must read it the same way.
        return {}
    if not isinstance(environment, dict):
        raise ManifestError(f"{field} must be a mapping")
    for key, value in environment.items():
        _require_string(key, f"{field} key")
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ManifestError(f"{field}[{key}] must be a scalar")
    return {str(key): str(value) for key, value in environment.items()}


def _compose_entry(entry, config, platform, display_name, consumed):
    """Render one manifest entry as the step the runner executes."""
    environment = _environment(entry, f"step {entry['id']}.environment")
    for key, variable in (entry.get("environment_from_config") or {}).items():
        block = config.get(key)
        if not block:
            raise ManifestError(
                f"step {entry['id']!r} reads {key!r} from the {platform} config, "
                "which does not define it"
            )
        if not isinstance(block, dict):
            raise ManifestError(
                f"step {entry['id']!r} reads {key!r} from the {platform} "
                f"config, which states it as {type(block).__name__} rather "
                "than the mapping the step is handed"
            )
        environment[variable] = json.dumps(block, separators=(",", ":"), sort_keys=True)
        consumed.add(key)
    declared = entry_metadata(entry, f"step {entry['id']!r}")
    step = {
        "id": entry["id"],
        "name": entry["name"].replace(DISPLAY_NAME_TOKEN, display_name),
        "command": entry["command"],
    }
    if declared["requires"]:
        step["requires"] = declared["requires"]
    if declared["timeout_minutes"] is not None:
        step["timeout_seconds"] = declared["timeout_minutes"] * 60
    if declared["resources"]:
        step["resources"] = declared["resources"]
    if environment:
        step["environment"] = environment
    return step


def _check_config_consumed(config, platform, steps, consumed):
    """Refuse expectations a config states that no running step reads.

    The other direction from ``_compose_entry``: data nobody checks is data
    nobody validates. Left unchecked, a platform can carry a whole expectation
    block and run against nothing.
    """
    declared = {
        key for entry in steps for key in (entry.get("environment_from_config") or {})
    }
    unread = sorted(key for key in declared if config.get(key) and key not in consumed)
    if unread:
        raise ManifestError(
            f"the {platform} config defines {', '.join(unread)}, which no step that "
            "runs on this platform consumes"
        )


def _prepare(manifest, config, platform, known):
    steps, _ids, _assigned = validate_manifest(manifest, known)
    if platform not in known:
        raise ManifestError(
            f"unknown platform {platform!r}; known platforms are {', '.join(known)}"
        )
    if not isinstance(config, dict):
        raise ManifestError("the platform config must be a mapping")
    display_name = _require_string(config.get("display_name"), "display_name")
    return steps, display_name


def compose(manifest, config, platform, known):
    """The `integration_tests` list for one platform.

    Order is the manifest's, never the config's, so two platforms that run the
    same steps run them in the same sequence.
    """
    steps, display_name = _prepare(manifest, config, platform, known)

    consumed: set[str] = set()
    composed = []
    for entry in steps:
        if platform not in _entry_platforms(entry, known):
            continue
        composed.append(_compose_entry(entry, config, platform, display_name, consumed))

    _check_config_consumed(config, platform, steps, consumed)

    if not composed:
        raise ManifestError(f"no steps apply to {platform!r}")
    return composed


def select(
    manifest,
    config,
    platform,
    known,
    level=DEFAULT_LEVEL,
    profile="pr",
    cases=DEFAULT_CASES,
    level_order=None,
):
    """The groups one plan runs on one platform, and why each other is skipped.

    Selection is subtractive and every subtraction is a manifest statement: a
    group left out because its level is deeper than the plan, because it
    declares other profiles, because its case class is not in this profile's
    set, or because something it requires was not selected. Execution order is
    still the manifest's -- ``requires`` only has to name an earlier id, which
    ``validate_manifest`` enforces -- so the returned list is a subsequence of
    ``compose()``'s.

    The skipped list is returned rather than logged because the plan carries it
    into the artifact: a group that did not run is a fact a reviewer checks,
    not one a CI log search has to reconstruct.
    """
    order = dict(level_order or LEVEL_ORDER)
    if level not in order:
        raise ManifestError(f"unknown level {level!r}")
    cases = _string_list(list(cases), "profile cases", CASE_CLASSES)
    steps, display_name = _prepare(manifest, config, platform, known)

    # A requirement is only a dependency where it also runs: `musa_runtime` is
    # declared for MUSA alone, so a group that requires it is unconstrained on
    # the other platforms rather than permanently blocked by a group they never
    # had.
    applicable = {
        entry["id"] for entry in steps if platform in _entry_platforms(entry, known)
    }

    chosen, skipped, selected_ids = [], [], set()
    consumed: set[str] = set()
    for entry in steps:
        if platform not in _entry_platforms(entry, known):
            continue
        step_id = entry["id"]
        declared = entry_metadata(entry, f"step {step_id!r}")
        reason = None
        if order[declared["level"]] > order[level]:
            reason = f"requires {declared['level']}, the plan runs {level}"
        elif declared["profiles"] is not None and profile not in declared["profiles"]:
            reason = f"not declared for the {profile} profile"
        elif not set(declared["cases"]) & set(cases):
            reason = f"not in the {profile} case set ({', '.join(cases)})"
        else:
            missing = [
                item
                for item in declared["requires"]
                if item in applicable and item not in selected_ids
            ]
            if missing:
                reason = f"dependency-blocked: {', '.join(missing)} was not selected"
        if reason is not None:
            skipped.append({"id": step_id, "name": entry["name"], "reason": reason})
            continue
        chosen.append(_compose_entry(entry, config, platform, display_name, consumed))
        selected_ids.add(step_id)

    _check_config_consumed(config, platform, steps, consumed)

    if not chosen:
        raise ManifestError(
            f"no steps apply to {platform!r} at {level} for the {profile} profile"
        )
    return chosen, skipped


def filter_groups(composed, wanted, platform):
    """Narrow a composed list to the ids a plan selected, in manifest order.

    The plan already chose these ids with ``select()``; this is the seam that
    hands that choice to the runner. It re-filters rather than trusting the
    order it was given, keeps the manifest's order, and refuses an id the
    manifest does not define *for this platform*: a plan naming a group this
    commit does not have describes a different commit, and running the
    remainder would report success for a selection that was never honoured.

    Exactness is deliberate in the other direction too. The result must be
    the same set the plan stated, so a filter that silently dropped an id is
    as wrong as one that silently added a group.
    """
    if not isinstance(wanted, list) or not all(
        isinstance(item, str) for item in wanted
    ):
        raise ManifestError("the selected groups must be a list of ids")
    if len(set(wanted)) != len(wanted):
        raise ManifestError("the selected groups repeat an id")
    available = {step["id"]: step for step in composed}
    unknown = [step_id for step_id in wanted if step_id not in available]
    if unknown:
        raise ManifestError(
            f"the plan selects {', '.join(unknown)} on {platform!r}, which the "
            "manifest does not define for this platform"
        )
    chosen_ids = set(wanted)
    kept = [step for step in composed if step["id"] in chosen_ids]
    dropped = [
        {
            "id": step["id"],
            "name": step["name"],
            "reason": "not selected by the plan",
        }
        for step in composed
        if step["id"] not in chosen_ids
    ]
    if not kept:
        raise ManifestError(
            f"the plan selects no group on {platform!r}; a plan that runs nothing "
            "cannot report success"
        )
    return kept, dropped


def coverage_report(assigned, known):
    """Lines describing how far the platforms still differ, worst first."""
    lines = []
    for step_id, platforms in assigned.items():
        missing = [platform for platform in known if platform not in platforms]
        if missing:
            lines.append(
                f"  {step_id}: {len(known) - len(missing)}/{len(known)} platforms "
                f"(no {', '.join(missing)})"
            )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--manifest", required=True, help="manifest as JSON")
    parser.add_argument("--platform-config", required=True, help="config as JSON")
    parser.add_argument(
        "--config-dir",
        default=".github/configs",
        help="directory the platform list is read from",
    )
    parser.add_argument(
        "--groups-file",
        default="",
        help="JSON array of ids the plan selected; omit to compose every group",
    )
    parser.add_argument(
        "--not-selected-output",
        default="",
        help="write the groups the filter dropped, with a reason, to this path",
    )
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(Path(args.manifest).read_text())
        config = json.loads(Path(args.platform_config).read_text())
        known = known_platforms(args.config_dir)
        steps, _ids, assigned = validate_manifest(manifest, known)
        composed = compose(manifest, config, args.platform, known)
        dropped = []
        if args.groups_file:
            wanted = json.loads(Path(args.groups_file).read_text())
            composed, dropped = filter_groups(composed, wanted, args.platform)
    except ManifestError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as error:
        print(f"::error::cannot read the manifest inputs: {error}", file=sys.stderr)
        return 1

    # Printed on every load, in every platform's run: the number is the
    # progress measure for making the platforms one pipeline instead of eight.
    variants = len(steps) - len(assigned)
    print(
        f"{args.platform}: {len(composed)} groups from {len(steps)} manifest entries "
        f"({variants} platform-restricted variant(s))",
        file=sys.stderr,
    )
    if dropped:
        print(
            f"{len(dropped)} group(s) the plan did not select on {args.platform}:",
            file=sys.stderr,
        )
        for group in dropped:
            print(f"  {group['id']}: {group['reason']}", file=sys.stderr)
    report = coverage_report(assigned, known)
    if report:
        print(
            f"{len(report)} step(s) do not run on every platform yet:",
            file=sys.stderr,
        )
        for line in report:
            print(line, file=sys.stderr)

    if args.not_selected_output:
        Path(args.not_selected_output).write_text(json.dumps(dropped, indent=2) + "\n")

    print(json.dumps(composed, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
