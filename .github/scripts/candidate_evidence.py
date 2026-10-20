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

"""Bind release acceptance to exact wheel bytes and fail closed on partial evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform as host_platform
import subprocess
import sys
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_installation(wheel_path):
    """Prove the installed package files are the candidate archive's bytes."""
    distribution = importlib.metadata.distribution("torch_fl")
    with zipfile.ZipFile(wheel_path) as wheel:
        members = [
            name
            for name in wheel.namelist()
            if name.startswith("torch_fl/") and not name.endswith("/")
        ]
        if not members:
            raise ValueError("candidate archive contains no torch_fl package")
        for name in members:
            installed = Path(distribution.locate_file(name))
            candidate_bytes = wheel.read(name)
            candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
            installed_sha = (
                hashlib.sha256(installed.read_bytes()).hexdigest()
                if installed.is_file()
                else "missing"
            )
            if installed_sha != candidate_sha:
                raise ValueError(
                    "installed package differs from candidate: "
                    f"{name} (installed={installed} installed_sha256={installed_sha} "
                    f"candidate_sha256={candidate_sha})"
                )


def verify(receipt, wheel, source_sha, platform, run_id):
    expected = {
        "source_sha": source_sha,
        "platform": platform,
        "run_id": run_id,
        "wheel_sha256": sha256(wheel),
        "wheel": wheel.name,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("candidate identity mismatch")
    if receipt.get("status") != "accepted" or receipt.get("profile") != "release":
        raise ValueError("candidate has no complete L4 release acceptance")


def finish(receipt, report, hardware, requirements):
    metadata = report["flagcicd"]
    if (
        report["exitcode"] != 0
        or metadata["complete"] is not True
        or metadata["source_ref"] != receipt["source_sha"]
        or metadata["source_repository"] != receipt["source_repository"]
        or metadata["platform"] != receipt["platform"]
    ):
        raise ValueError(
            "candidate execution is incomplete or belongs to another source"
        )
    if any(
        report["summary"].get(key, 0)
        for key in ("failed", "error", "skipped", "xfailed", "xpassed")
    ):
        raise ValueError(
            "release acceptance contains failures or unapproved skipped cases"
        )
    execution = metadata.get("execution") or {}
    groups = execution.get("groups", [])
    if not execution.get("complete") or execution.get("expected") != [
        g["name"] for g in groups
    ]:
        raise ValueError("candidate group inventory is incomplete")
    if not groups or any(group["status"] != "passed" for group in groups):
        raise ValueError("candidate groups did not all pass")
    categories = {group["category"] for group in groups}
    if not set(requirements["required_categories"]) <= categories:
        raise ValueError("candidate is missing mandatory acceptance categories")
    if hardware.get("accelerator_count", 0) < 1 or hardware.get("sdk_version") in (
        None,
        "",
        "unknown",
    ):
        raise ValueError("candidate hardware/SDK identity is unavailable")
    return {
        **receipt,
        "status": "accepted",
        "test_summary": report["summary"],
        "hardware": hardware,
        "execution": execution,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "finish", "verify"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hardware", type=Path)
    args = parser.parse_args()
    identity = {
        "source_sha": os.environ["GITHUB_SHA"],
        "platform": args.platform,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "wheel_sha256": sha256(args.wheel),
        "wheel": args.wheel.name,
    }
    if args.mode == "record":
        verify_installation(args.wheel)
        with zipfile.ZipFile(args.wheel) as wheel:
            compatibility = json.loads(wheel.read("torch_fl/compatibility.json"))
        receipt = {
            **identity,
            "schema_version": "flagcicd.candidate/v1",
            "status": "pending",
            "source_repository": os.environ["GITHUB_REPOSITORY"],
            "profile": "release",
            "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
            "compatibility": compatibility,
            "python": host_platform.python_version(),
            "image": os.environ.get("CANDIDATE_IMAGE", ""),
            "dependencies": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True, timeout=120
            ).splitlines(),
        }
    else:
        receipt = json.loads(args.receipt.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "candidate bytes or source identity changed during acceptance"
            )
        if args.mode == "verify":
            verify(
                receipt,
                args.wheel,
                identity["source_sha"],
                args.platform,
                identity["run_id"],
            )
            return
        verify_installation(args.wheel)
        requirements = json.loads(
            (
                Path(__file__).resolve().parents[2] / ".flagcicd/test-tiers.json"
            ).read_text()
        )["release"]
        receipt = finish(
            receipt,
            json.loads(args.report.read_text()),
            json.loads(args.hardware.read_text()),
            requirements,
        )
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
