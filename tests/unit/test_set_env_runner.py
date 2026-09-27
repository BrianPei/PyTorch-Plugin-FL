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

"""The set_env entrypoint: one runner, per-platform hooks, thin wrappers.

`.github/scripts/set_env.sh --platform <p>` owns the boilerplate every platform
shares and sources `hooks/set_env_<p>.sh`, which is the platform-specific part.
The old `set_env_<p>.sh` files are thin wrappers around the runner, and the
named integration workflows call the runner directly. This pins that shape: the
runner dispatches, every platform has a hook and a wrapper, and the workflows
use the parameterized entrypoint.

The runner is exercised with `--help`/an unknown platform, which return before
any provisioning runs.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / ".github" / "scripts"
RUNNER = SCRIPTS / "set_env.sh"
HOOKS = SCRIPTS / "hooks"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

PLATFORMS = ["cuda", "ascend", "dcu", "gcu", "metax", "musa", "ppu"]

#: The named workflows that provision a platform themselves.
NAMED_WORKFLOWS = ["cuda", "ascend", "dcu", "metax", "musa", "ppu"]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        capture_output=True,
        text=True,
        env={"CI_STAGE": "build", "PATH": "/usr/bin:/bin"},
    )


def test_runner_rejects_a_missing_platform():
    done = _run()
    assert done.returncode == 2
    assert "--platform is required" in done.stderr


def test_runner_rejects_an_unknown_platform():
    done = _run("--platform", "bogus")
    assert done.returncode == 2
    assert "unknown platform 'bogus'" in done.stderr


def test_runner_rejects_an_unknown_argument():
    done = _run("--nope")
    assert done.returncode == 2
    assert "unknown argument" in done.stderr


def test_every_platform_has_a_hook():
    for platform in PLATFORMS:
        assert (HOOKS / f"set_env_{platform}.sh").is_file(), platform


def test_every_wrapper_routes_to_the_runner():
    for platform in PLATFORMS:
        text = (SCRIPTS / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        assert "set_env.sh" in text and f"--platform {platform}" in text, platform


def test_named_workflows_use_the_parameterized_entrypoint():
    for platform in NAMED_WORKFLOWS:
        text = (WORKFLOWS / f"integration-test-{platform}.yml").read_text(
            encoding="utf-8"
        )
        assert f"bash .github/scripts/set_env.sh --platform {platform}" in text, (
            platform
        )


def test_config_driven_path_routes_through_a_wrapper():
    """The configs name the wrapper, which routes to the runner.

    `all-tests-common.yml` validates that the config's setup_script is a file and
    runs it with no arguments, so the config keeps pointing at the wrapper; the
    wrapper is the parameterized call.
    """
    for platform in PLATFORMS:
        text = (REPO_ROOT / ".github" / "configs" / f"{platform}.yml").read_text(
            encoding="utf-8"
        )
        assert f"setup_script: .github/scripts/set_env_{platform}.sh" in text, platform
