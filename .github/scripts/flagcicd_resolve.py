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

"""Resolve one GitHub event into the request every FlagCICD entry consumes.

Two workflows accept work: ``ci.yml`` for pull_request / push / schedule, and
``flagcicd-entry.yml`` for manual and external dispatch. They must not each
decide what "the source" means: a plan is only comparable to another plan when
both sides resolved the same fields. This module is that single reading. It
turns an event payload into ``flagcicd.request/v1`` and every consumer
(``flagcicd_plan.py``, ``flagcicd_report.py``) takes its inputs from there.

Three commits are tracked separately, because they answer different questions:

``source_repository`` / ``source_sha``
    the code under test. For a pull_request that is the head commit, not the
    synthetic merge commit: the entry workflow tests the head, and two entries
    for one PR must resolve the same source or their reports cannot be
    compared.

``base_sha``
    the comparison point for ``git diff base...head``. Empty means "this event
    has no comparison commit" -- the planner reads that conservatively, never
    as "nothing changed".

``control_sha``
    the commit whose workflows and scripts produced this run
    (``github.workflow_sha``). A result is attributable to control code even
    when that code came from the base branch and the tested source did not.

The dispatch format is unchanged: ``pr_number``, ``checkout_ref``, ``base_ref``,
``platforms``, ``test_level``, ``task_id`` and ``parent_run_id`` keep their
meaning. ``selection`` and ``profile`` are added because ``platforms=all``
currently carries two meanings -- "every platform" and "every platform the
diff affects" -- and a plan has to state which one it was.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import yaml


SCHEMA = "flagcicd.request/v1"
LEVELS = ("auto", "L0", "L1", "L2", "L3", "L4")
SHA_RE = re.compile(r"[0-9a-f]{40}")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
PLATFORMS_RE = re.compile(r"all|[a-z0-9][a-z0-9_-]*(?:,[a-z0-9][a-z0-9_-]*)*")
ZERO_SHA = "0" * 40


def text(value, field):
    if (
        not isinstance(value, str)
        or len(value) > 256
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(
            f"{field} must be a single-line string of at most 256 characters"
        )
    return value


def api(path):
    result = subprocess.run(
        ["gh", "api", path], check=True, capture_output=True, text=True, timeout=60
    )
    return json.loads(result.stdout)


def commit(repository, ref):
    sha = api(f"repos/{repository}/commits/{quote(ref, safe='')}")["sha"]
    if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
        raise ValueError("GitHub did not return a full commit SHA")
    return sha


def sha(value, field, allow_empty=False):
    if value in ("", None):
        if allow_empty:
            return ""
        raise ValueError(f"{field} must be an immutable commit SHA")
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ValueError(f"{field} must be an immutable commit SHA")
    return value


def repository(value, field):
    if not isinstance(value, str) or not REPOSITORY_RE.fullmatch(value):
        raise ValueError(f"{field} is unavailable")
    return value


def labels(value):
    """PR labels as plain names; an unreadable label is dropped, not guessed."""
    if not isinstance(value, list):
        return []
    names = []
    for label in value:
        name = label.get("name") if isinstance(label, dict) else label
        if isinstance(name, str) and name.strip() and len(name) <= 100:
            names.append(name.strip())
    return names


def normalized(*, profile, selection, platforms, event_labels, contract, **fields):
    """Validate one request against the contract and fill in its defaults."""
    if profile not in contract["profiles"]:
        raise ValueError(
            f"unknown profile {profile!r}; declare it under profiles in "
            ".flagcicd/project.yaml"
        )
    if selection not in ("affected", "all"):
        raise ValueError("selection must be affected or all")
    platforms = platforms.strip() if isinstance(platforms, str) else ""
    if not PLATFORMS_RE.fullmatch(platforms):
        raise ValueError("platforms must be all or a comma-separated list of names")
    if selection == "all" and platforms != "all":
        raise ValueError("selection=all cannot be combined with a platform list")
    requested_level = fields.pop("requested_level", "auto")
    if requested_level not in LEVELS:
        raise ValueError(
            f"supported test levels are {', '.join(LEVELS)}; got {requested_level!r}"
        )
    return {
        "schema_version": SCHEMA,
        "profile": profile,
        "selection": selection,
        "platforms": platforms,
        "labels": event_labels,
        "requested_level": requested_level,
        "control_sha": fields.pop("control_sha", ""),
        "task_id": fields.pop("task_id", ""),
        "parent_run_id": fields.pop("parent_run_id", ""),
        **fields,
    }


def event_request(event, event_name, repo, event_sha, control_sha, contract):
    """Resolve a native GitHub event into a normalized request.

    ``pull_request`` is read from the payload. ci.yml used to re-fetch the PR
    through the API; the payload already carries the same immutable head and
    base commits, and resolving them here removes a token dependency from the
    step that decides what will be tested.
    """
    if event_name == "pull_request":
        payload = event.get("pull_request") or {}
        head = payload.get("head") or {}
        base = payload.get("base") or {}
        return normalized(
            profile="pr",
            selection="affected",
            platforms="all",
            event_labels=labels(payload.get("labels")),
            contract=contract,
            event_name=event_name,
            pr_number=payload.get("number", ""),
            source_repository=repository(
                (head.get("repo") or {}).get("full_name", ""),
                "pull request head repository",
            ),
            source_sha=sha(head.get("sha"), "pull request head SHA"),
            base_sha=sha(base.get("sha"), "pull request base SHA"),
            control_sha=control_sha,
        )
    if event_name == "push":
        before = event.get("before") or ""
        # A new branch reports the zero object as `before`; there is no
        # comparison commit, and the planner must know that rather than diff
        # against nothing and call the result clean.
        base_sha = "" if before == ZERO_SHA else sha(before, "push base SHA")
        return normalized(
            profile="pr",
            selection="affected" if base_sha else "all",
            platforms="all",
            event_labels=[],
            contract=contract,
            event_name=event_name,
            ref=event.get("ref") or "",
            source_repository=repository(
                (event.get("repository") or {}).get("full_name", repo),
                "source repository",
            ),
            source_sha=sha(event.get("after"), "pushed commit SHA"),
            base_sha=base_sha,
            control_sha=control_sha,
        )
    if event_name == "schedule":
        # The cron in the payload is the only fact that distinguishes one
        # scheduled profile from another: the daily run and the weekly run are
        # different budgets with different case sets, and a hardcoded profile
        # here would silently give the weekly schedule the daily one. The
        # contract's `schedules` block is the single place a cron becomes a
        # profile; a cron it does not declare is a workflow/contract
        # disagreement, and refusing names both sides.
        cron = event.get("schedule")
        schedules = contract.get("schedules") or {}
        if cron not in schedules:
            raise ValueError(
                f"scheduled event {cron!r} is not declared under schedules in "
                f".flagcicd/project.yaml; declared: {', '.join(sorted(schedules))}"
            )
        return normalized(
            profile=schedules[cron],
            selection="all",
            platforms="all",
            event_labels=[],
            contract=contract,
            event_name=event_name,
            source_repository=repository(repo, "source repository"),
            source_sha=sha(event_sha, "scheduled commit SHA"),
            base_sha="",
            control_sha=control_sha,
        )
    raise ValueError(f"{event_name} has no automatic request")


def dispatch_request(event, event_name, repo, event_sha, control_sha, contract):
    """Resolve workflow_dispatch / repository_dispatch:flagcicd requests."""
    if event_name == "workflow_dispatch":
        request = event.get("inputs") or {}
    elif event_name == "repository_dispatch" and event.get("action") == "flagcicd":
        request = event.get("client_payload") or {}
    else:
        raise ValueError(
            "only workflow_dispatch or repository_dispatch:flagcicd is supported"
        )
    if not isinstance(request, dict):
        raise ValueError("dispatch request must be an object")
    number = str(request.get("pr_number", ""))
    checkout_ref = text(request.get("checkout_ref", ""), "checkout_ref")
    base_ref = text(request.get("base_ref", ""), "base_ref")
    if number and (checkout_ref or base_ref):
        raise ValueError("pr_number cannot be combined with checkout_ref or base_ref")
    source_repository = repo
    if number:
        if not re.fullmatch(r"[1-9][0-9]*", number):
            raise ValueError("pr_number must be a positive integer")
        pr = api(f"repos/{repo}/pulls/{number}")
        if pr.get("state") != "open":
            raise ValueError("the requested PR is not open")
        source_repository = (pr["head"].get("repo") or {}).get("full_name", "")
        source_sha = sha(pr["head"]["sha"], "pull request head SHA")
        base_sha = sha(pr["base"]["sha"], "pull request base SHA")
    else:
        source_sha = commit(repo, checkout_ref or event_sha)
        if not base_ref:
            base_ref = api(f"repos/{repo}")["default_branch"]
        base_sha = commit(repo, base_ref)
    platforms = request.get("platforms", "all")
    if isinstance(platforms, list):
        if not platforms or not all(isinstance(item, str) for item in platforms):
            raise ValueError(
                "platforms must be all, a comma-separated list, or a JSON array"
            )
        # An array names one platform per element. A comma inside an element
        # is a string list in disguise, and joining it would quietly accept
        # "cuda,ppu" as two platforms when the caller wrote one entry.
        if any("," in item or not item.strip() for item in platforms):
            raise ValueError("each platform in an array must be a single name")
        platforms = ",".join(platforms)
    profile = text(request.get("profile") or "pr", "profile")
    # A dispatch form submits every input it declares, so an untouched field
    # arrives as "" rather than as absent. Empty means "use the profile's
    # default"; only an explicit value is validated.
    return normalized(
        profile=profile,
        selection=text(
            request.get("selection") or contract["profiles"][profile]["selection"],
            "selection",
        ),
        platforms=platforms,
        event_labels=[],
        contract=contract,
        event_name=event_name,
        pr_number=number,
        source_repository=repository(source_repository, "source repository"),
        source_sha=source_sha,
        base_sha=base_sha,
        control_sha=control_sha,
        requested_level=text(request.get("test_level", "auto"), "test_level"),
        task_id=text(request.get("task_id", ""), "task_id"),
        parent_run_id=text(request.get("parent_run_id", ""), "parent_run_id"),
    )


def resolve(event, event_name, repo, event_sha, control_sha, contract):
    if event_name in ("workflow_dispatch", "repository_dispatch"):
        return dispatch_request(
            event, event_name, repo, event_sha, control_sha, contract
        )
    return event_request(event, event_name, repo, event_sha, control_sha, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(".flagcicd/project.yaml"),
        help="capability contract read from the trusted control checkout",
    )
    args = parser.parse_args()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    request = resolve(
        event,
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_SHA"],
        os.environ.get("GITHUB_WORKFLOW_SHA", ""),
        yaml.safe_load(args.contract.read_text()),
    )
    args.output.write_text(json.dumps(request, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key in (
            "source_repository",
            "source_sha",
            "base_sha",
            "control_sha",
            "profile",
            "selection",
        ):
            output.write(f"{key}={request[key]}\n")
        output.write(f"pr_number={request.get('pr_number', '')}\n")


if __name__ == "__main__":
    main()
