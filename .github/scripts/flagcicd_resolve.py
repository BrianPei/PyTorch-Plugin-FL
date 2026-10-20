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

"""Normalize CI and dispatch events before checking out immutable source data."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote


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
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GitHub did not return a full commit SHA")
    return sha


def ci_request(event, event_name):
    """Preserve each automatic event's scheduling and checkout semantics."""
    if event_name == "pull_request":
        pr = event["pull_request"]
        return {
            "test_level": "L1",
            "profile": "auto",
            "selection": "affected",
            "pr_labels": [label["name"] for label in pr.get("labels", [])],
        }
    if event_name == "push":
        return {"profile": "pr", "selection": "all"}
    if event_name == "schedule":
        return {"profile": "nightly", "selection": "all"}
    if event_name == "workflow_dispatch":
        inputs = event.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise ValueError("dispatch request must be an object")
        return {"test_level": "L2", **inputs}
    raise ValueError("unsupported CI event")


def resolve(event, event_name, repository, event_sha, entry="external", control_sha=""):
    if entry == "ci":
        request = ci_request(event, event_name)
    elif event_name == "workflow_dispatch":
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
    source_repository = repository
    target_kind = "commit"
    if entry == "ci" and event_name == "pull_request":
        pr = event["pull_request"]
        source_sha = event_sha
        base_sha = pr["base"]["sha"]
        target_kind = "merge"
    elif entry == "ci" and event_name in ("push", "schedule"):
        source_sha = event_sha
        previous = event.get("before", "")
        base_sha = (
            previous
            if isinstance(previous, str)
            and re.fullmatch(r"[0-9a-f]{40}", previous)
            and previous != "0" * 40
            else source_sha
        )
    elif number:
        if not re.fullmatch(r"[1-9][0-9]*", number):
            raise ValueError("pr_number must be a positive integer")
        pr = api(f"repos/{repository}/pulls/{number}")
        if pr.get("state") != "open":
            raise ValueError("the requested PR is not open")
        source_repository = (pr["head"].get("repo") or {}).get("full_name", "")
        source_sha = pr["head"]["sha"]
        base_sha = pr["base"]["sha"]
        target_kind = "head"
    else:
        source_sha = commit(repository, checkout_ref or event_sha)
        if not base_ref:
            base_ref = api(f"repos/{repository}")["default_branch"]
        base_sha = commit(repository, base_ref)
    if not isinstance(source_repository, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source_repository
    ):
        raise ValueError("source repository is unavailable")
    diff_sha = (
        event["pull_request"]["head"]["sha"]
        if entry == "ci" and event_name == "pull_request"
        else source_sha
    )
    for sha in (source_sha, base_sha, diff_sha):
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("source and base must be immutable commit SHAs")
    if control_sha and not re.fullmatch(r"[0-9a-f]{40}", control_sha):
        raise ValueError("control code must use an immutable commit SHA")
    requested_level = text(request.get("test_level", "auto"), "test_level")
    if requested_level not in ("auto", "L0", "L1", "L2", "L3"):
        raise ValueError("supported test levels are auto, L0, L1, L2 and L3")
    profile = text(request.get("profile", "auto"), "profile")
    if profile not in ("auto", "pr", "pr-risk", "nightly", "weekly"):
        raise ValueError("unsupported execution profile; L4 uses the release workflow")
    platforms = request.get("platforms", "all")
    legacy_all = isinstance(platforms, str) and platforms.strip() == "all"
    selection = request.get("selection", "legacy")
    if selection == "legacy":
        selection = "all" if entry == "ci" else ("affected" if legacy_all else "all")
        origin = "legacy_ci_manual" if entry == "ci" else "legacy_external"
    else:
        origin = "explicit"
    if entry == "ci" and event_name != "workflow_dispatch":
        origin = "event_policy"
    if selection not in ("affected", "all"):
        raise ValueError("selection must be affected or all")
    return {
        "profile": profile,
        "source_repository": source_repository,
        "source_sha": source_sha,
        "base_sha": base_sha,
        "diff_sha": diff_sha,
        "control_sha": control_sha,
        "target_kind": target_kind,
        "selection": selection,
        "selection_origin": origin,
        "pr_labels": request.get("pr_labels", []) if entry == "ci" else [],
        "platforms": platforms,
        "requested_level": requested_level,
        "task_id": text(request.get("task_id", ""), "task_id"),
        "parent_run_id": text(request.get("parent_run_id", ""), "parent_run_id"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry", choices=("external", "ci"), default="external")
    args = parser.parse_args()
    request = resolve(
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_SHA"],
        args.entry,
        os.environ.get("CONTROL_SHA", ""),
    )
    args.output.write_text(json.dumps(request, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key in ("source_repository", "source_sha", "base_sha"):
            output.write(f"{key}={request[key]}\n")
        output.write("request=" + json.dumps(request, separators=(",", ":")) + "\n")
        output.write(f"repository={request['source_repository']}\n")
        output.write(f"ref={request['source_sha']}\n")


if __name__ == "__main__":
    main()
