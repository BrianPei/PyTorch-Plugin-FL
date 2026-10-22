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

"""Hold the project contract, the workflows it names and the run gate together.

``project.yaml`` used to describe a gate no code read: the entry workflow ran its
own job list and ``flagcicd_report.py`` decided completeness from a hardcoded set.
These tests make every declared gate a verifiable one, so the contract cannot drift
away from what CI actually enforces.
"""

import re
import sys
from pathlib import Path
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
from contract_schema import validate_project  # noqa: E402
from flagcicd_report import aggregate  # noqa: E402
from test_tiers import acceptance_policy, environment_contract, policy  # noqa: E402

sys.path.pop(0)

CONTRACT = ROOT / ".flagcicd/project.yaml"


def load_contract():
    return yaml.safe_load(CONTRACT.read_text())


def expected_jobs(level):
    """Recover the gate for a level from the code that enforces it.

    ``aggregate`` reports every job it expected to succeed; reading those errors
    keeps this test from restating a constant that would then be free to drift.
    """
    plan = {"expected_platforms": [], "test_level": level}
    with tempfile.TemporaryDirectory() as directory:
        report = aggregate(plan, {}, Path(directory), 1)
    pattern = re.compile(r"^(\S+): expected success, received")
    return {match.group(1) for match in map(pattern.match, report["errors"]) if match}


def resolve(document, path):
    value = document
    for name in path.split("."):
        value = value[name]
    return value


def test_project_contract_satisfies_its_schema():
    validate_project(load_contract())


def test_declared_checks_exist_in_their_workflows():
    contract = load_contract()
    for check_id, check in contract["checks"].items():
        workflow = ROOT / check["workflow"]
        assert workflow.is_file(), f"{check_id}: missing {check['workflow']}"
        jobs = yaml.safe_load(workflow.read_text()).get("jobs", {})
        assert check["job"] in jobs, (
            f"{check_id}: {check['workflow']} has no job {check['job']}"
        )


def test_level_checks_match_the_run_gate():
    contract = load_contract()
    for level, definition in contract["levels"].items():
        declared = {
            contract["checks"][check_id]["entry_job"]
            for check_id in definition["checks"]
        }
        assert declared == expected_jobs(level), (
            f"{level}: contract declares {sorted(declared)}, "
            f"the run gate requires {sorted(expected_jobs(level))}"
        )


def test_adapter_paths_exist_and_parse():
    contract = load_contract()
    for name, relative in contract["adapter"].items():
        assert (ROOT / relative).is_file(), f"{name}: missing {relative}"
    # The loaders reject a schema the core does not implement, so a successful read
    # is also the version check.
    assert environment_contract()["schema_version"] == "flagcicd.environment/v1"
    assert acceptance_policy()["schema_version"] == "flagcicd.acceptance/v1"
    assert policy()["schema_version"] == "flagcicd.tiers/v1"


def test_acceptance_extras_resolve():
    document = policy()
    for extra in acceptance_policy()["extras"]:
        assert (ROOT / extra["path"]).is_file(), (
            f"{extra['id']}: missing {extra['path']}"
        )
        # ``select`` looks the extra up as a tier case; a missing one is a KeyError.
        assert extra["id"] in document["cases"], f"{extra['id']}: no tier case"
        if "platforms_source" in extra:
            assert resolve(document, extra["platforms_source"]), (
                f"{extra['id']}: empty {extra['platforms_source']}"
            )
