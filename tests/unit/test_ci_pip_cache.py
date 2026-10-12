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

"""Replay cache-action outputs, including download errors reported as success."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = [
    "build-wheel-common.yml",
    "integration-tests-common.yml",
    *[
        f"integration-test-{p}.yml"
        for p in ("ascend", "cuda", "dcu", "metax", "musa", "ppu")
    ],
]


def _steps(workflow):
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return next(job["steps"] for job in document["jobs"].values() if "steps" in job)


def _condition(expression, states, cancelled=False):
    """Evaluate the workflow's boolean expression against recorded action outputs."""

    def output(match):
        step, field, name = match.groups()
        state = states.get(step, {})
        value = state.get("outputs", {}).get(name, "") if name else state.get(field, "")
        return repr(value)

    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = re.sub(
        r"steps\.([a-z-]+)\.(outcome|outputs\.([a-z-]+))", output, expression
    )
    expression = expression.replace("cancelled()", repr(cancelled))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression)
    return eval(expression.strip(), {"__builtins__": {}}, {})


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize(
    "lookup_outcome,primary,hit,matched,restore_outcome,restored,cancelled,download,save",
    [
        pytest.param(
            "success", "new", "", "", "skipped", "", False, False, True, id="cold-miss"
        ),
        pytest.param(
            "failure",
            "new",
            "",
            "",
            "skipped",
            "",
            False,
            False,
            False,
            id="lookup-failed",
        ),
        pytest.param(
            "failure",
            "",
            "",
            "",
            "skipped",
            "",
            False,
            False,
            False,
            id="lookup-timeout",
        ),
        pytest.param(
            "success", "", "", "", "skipped", "", False, False, False, id="missing-key"
        ),
        pytest.param(
            "success",
            "new",
            "true",
            "new",
            "success",
            "new",
            False,
            True,
            False,
            id="exact-hit",
        ),
        pytest.param(
            "success",
            "new",
            "true",
            "new",
            "success",
            "",
            False,
            True,
            False,
            id="cuda-swallowed-timeout",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "success",
            "old",
            False,
            True,
            True,
            id="fallback-restored",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "success",
            "",
            False,
            True,
            False,
            id="gcu-swallowed-timeout",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "failure",
            "",
            False,
            True,
            False,
            id="transfer-step-timeout",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "skipped",
            "",
            False,
            True,
            False,
            id="transfer-skipped",
        ),
        pytest.param(
            "success",
            "new",
            "",
            "",
            "skipped",
            "",
            True,
            False,
            False,
            id="cancelled-cold-miss",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "success",
            "old",
            True,
            True,
            False,
            id="cancelled-fallback",
        ),
    ],
)
def test_cache_transfer_decisions(
    workflow,
    lookup_outcome,
    primary,
    hit,
    matched,
    restore_outcome,
    restored,
    cancelled,
    download,
    save,
):
    steps = _steps(workflow)
    lookup = next(step for step in steps if step.get("id") == "pip-cache-lookup")
    restore = next(step for step in steps if step.get("id") == "pip-cache")
    upload = next(step for step in steps if step.get("uses") == "actions/cache/save@v4")
    states = {
        "pip-cache-lookup": {
            "outcome": lookup_outcome,
            "outputs": {
                "cache-primary-key": primary,
                "cache-hit": hit,
                "cache-matched-key": matched,
            },
        },
        "pip-cache": {
            "outcome": restore_outcome,
            "outputs": {"cache-matched-key": restored},
        },
    }
    assert lookup["with"]["lookup-only"] is True
    assert _condition(restore["if"], states, cancelled) is download
    assert _condition(upload["if"], states, cancelled) is save
    assert (
        upload["with"]["key"]
        == "${{ steps.pip-cache-lookup.outputs.cache-primary-key }}"
    )


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_cache_failure_budgets_do_not_gate_installation(workflow):
    for step in _steps(workflow):
        if step.get("uses", "").startswith("actions/cache/"):
            assert step["continue-on-error"] is True
            maximum = 3 if step["uses"] == "actions/cache/save@v4" else 2
            assert 0 < step["timeout-minutes"] <= maximum
            if step.get("id") == "pip-cache":
                assert step["env"]["SEGMENT_DOWNLOAD_TIMEOUT_MINS"] == "1"
        elif "run" in step:
            assert "pip-cache" not in step.get("if", "")


def test_common_build_and_integration_share_the_dependency_key():
    keys = []
    for workflow in WORKFLOWS[:2]:
        lookup = next(
            step for step in _steps(workflow) if step.get("id") == "pip-cache-lookup"
        )
        keys.append(lookup["with"]["key"])
        assert "set_env_*.sh" not in keys[-1]
        assert "inputs.setup_platform" in keys[-1]
    assert keys[0] == keys[1]


def test_cuda_caches_cpu_torch_without_the_vendor_wheel_set():
    hook = (ROOT / ".github/scripts/hooks/set_env_cuda.sh").read_text()
    assert re.search(r"^install_cpu_torch\s*$", hook, re.M)
    assert re.search(
        r"pip_retry --no-cache-dir --index-url \"\$VENDOR_TORCH_INDEX_URL\"", hook
    )
