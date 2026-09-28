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

"""Exercise MUSA environment reuse without hardware, downloads or package installs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = REPO_ROOT / ".github/scripts/lib/set_env_common.sh"
COMMON = COMMON_PATH.read_text()
SCRIPT = (REPO_ROOT / ".github/scripts/hooks/set_env_musa.sh").read_text()


def shell_function(name, script=SCRIPT):
    start = script.index(f"{name}() {{")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def run_shell(body, **environment):
    env = os.environ.copy()
    env.update(PYTHONPATH="", PYTHONNOUSERSITE="1")
    env.update(environment)
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + body],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "condition,usable",
    [
        ("isolated", True),
        ("system_packages", False),
        ("conflicting_config", False),
        ("vendor_module", False),
        ("orphan_vendor_metadata", False),
        ("missing_pip", False),
        ("broken_pip", False),
    ],
)
def test_venv_reuse_requires_isolation(tmp_path, condition, usable):
    root = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
    )
    site = next((root / "lib").glob("python*/site-packages"))
    if condition != "missing_pip":
        (site / "pip.py").write_text(
            "raise SystemExit(1)" if condition == "broken_pip" else ""
        )
    if condition == "system_packages":
        config = root / "pyvenv.cfg"
        config.write_text(config.read_text().replace("= false", "= true"))
    if condition == "conflicting_config":
        config = root / "pyvenv.cfg"
        config.write_text(config.read_text() + "include-system-site-packages = true\n")
    if condition == "vendor_module":
        (site / "torch_musa.py").touch()
    if condition == "orphan_vendor_metadata":
        dist = site / "torch_musa-2.9.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Name: torch_musa\nVersion: 2.9.1\n")

    result = run_shell(
        shell_function("venv_is_usable", COMMON)
        + "\n"
        + shell_function("musa_venv_is_isolated")
        + "\nmusa_venv_is_isolated",
        VENV_ROOT=str(root),
        VENV_PYTHON=str(root / "bin/python"),
    )
    assert (result.returncode == 0) == usable, result.stdout + result.stderr


@pytest.mark.parametrize(
    "condition,reuse",
    [
        ("same_version", True),
        ("different_version", False),
        ("missing_module", False),
        ("missing_metadata", False),
    ],
)
def test_flaggems_reuse_requires_published_version(tmp_path, condition, reuse):
    dist = tmp_path / "flag_gems-5.4.0.dist-info"
    dist.mkdir()
    if condition != "missing_metadata":
        version = "5.3.0" if condition == "different_version" else "5.4.0"
        (dist / "METADATA").write_text(f"Name: flag_gems\nVersion: {version}\n")
    if condition != "missing_module":
        (tmp_path / "flag_gems.py").touch()

    result = run_shell(
        "pip_retry() {\n"
        '  printf "INSTALL_REQUESTED %s\\n" "$*"\n'
        '  printf "Name: flag_gems\\nVersion: 5.4.0\\n" > "$STUB_METADATA"\n'
        '  touch "$STUB_MODULE"\n'
        "}\n"
        + shell_function("flag_gems_installed", COMMON)
        + "\n"
        + shell_function("install_flag_gems", COMMON)
        + "\ninstall_flag_gems",
        VENV_PYTHON=sys.executable,
        PYTHONPATH=str(tmp_path),
        FLAGGEMS_VERSION="5.4.0",
        FLAGGEMS_INDEX_URL="https://example.invalid/mthreads/simple",
        STUB_METADATA=str(dist / "METADATA"),
        STUB_MODULE=str(tmp_path / "flag_gems.py"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("INSTALL_REQUESTED" not in result.stdout) == reuse
    if not reuse:
        assert "--only-binary=:all:" in result.stdout
        assert "flag-gems===5.4.0" in result.stdout
        assert "git+" not in result.stdout


def test_flaggems_rejects_an_unusable_installed_wheel(tmp_path):
    result = run_shell(
        "pip_retry() { return 0; }\n"
        + shell_function("flag_gems_installed", COMMON)
        + "\n"
        + shell_function("install_flag_gems", COMMON)
        + "\ninstall_flag_gems",
        VENV_PYTHON=sys.executable,
        PYTHONPATH=str(tmp_path),
        FLAGGEMS_VERSION="0.0.0-test-missing",
        FLAGGEMS_INDEX_URL="https://example.invalid/simple",
    )
    assert result.returncode != 0
    assert "not importable after wheel installation" in result.stdout


@pytest.mark.parametrize("qwen", ["0", "1"])
def test_integration_keeps_bert_and_gates_tokenizer_deps(qwen):
    # Execute the dependency block with an offline recording pip helper.
    start = SCRIPT.index('if [[ "$CI_STAGE" == "integration" ]]; then')
    end = SCRIPT.index("\nexport VIRTUAL_ENV=", start)
    result = run_shell(
        'pip_retry() { printf "%s\\n" "$@"; }\n' + SCRIPT[start:end],
        CI_STAGE="integration",
        PIP_INDEX_URL_ARG="https://example.invalid/simple",
        TORCH_FL_INSTALL_QWEN_DEPS=qwen,
    )
    assert result.returncode == 0, result.stderr
    assert "transformers>=4.51,<5" in result.stdout
    assert "PyYAML==6.0.1" in result.stdout
    for package in ("sentencepiece", "tiktoken", "protobuf"):
        assert (package in result.stdout) == (qwen == "1")


@pytest.mark.parametrize(
    "version,cuda,usable",
    [
        ("2.10.0+cpu", None, True),
        ("2.9.1+cpu", None, False),
        ("2.10.0", None, False),
        ("2.10.0+cpu", "12.8", False),
    ],
)
def test_cpu_torch_probe_requires_the_pinned_cpu_build(tmp_path, version, cuda, usable):
    (tmp_path / "torch.py").write_text(
        "import os\n"
        "assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] == '0'\n"
        f"__version__ = {version!r}\n"
        f"class version: cuda = {cuda!r}\n"
    )
    result = run_shell(
        shell_function("musa_cpu_torch_is_usable")
        + '\nmusa_cpu_torch_is_usable\necho "AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD"',
        VENV_PYTHON=sys.executable,
        CPU_TORCH_VERSION="2.10.0",
        PYTHONPATH=str(tmp_path),
        TORCH_DEVICE_BACKEND_AUTOLOAD="1",
    )
    assert (result.returncode == 0) == usable, result.stdout + result.stderr
    if usable:
        assert "AUTOLOAD=1" in result.stdout


@pytest.mark.parametrize("cache", ["0", "1"])
def test_shared_pip_retry_honors_musa_cache_setting(tmp_path, cache):
    recorder = tmp_path / "python"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    recorder.chmod(0o755)
    result = run_shell(
        shell_function("pip_retry", COMMON) + "\npip_retry package-for-test",
        VENV_PYTHON=str(recorder),
        PIP_RETRY_NO_CACHE=cache,
    )
    assert result.returncode == 0, result.stderr
    assert ("--no-cache-dir" in result.stdout) == (cache == "1")


@pytest.mark.parametrize(
    "scenario",
    [
        "fresh",
        "reused",
        "prebuilt",
        "invalid_local",
        "invalid_prebuilt",
        "stale_prebuilt",
        "noncpu_prebuilt",
    ],
)
@pytest.mark.parametrize("stage", ["build", "integration"])
def test_isolated_setup_reconciles_existing_and_fresh_environments(
    tmp_path, scenario, stage
):
    # Run the complete Python setup section, but replace package installs and
    # bootstrap creation with offline stubs. Real venvs still test isolation.
    root = tmp_path / "venv"
    marker = root / "keep-existing-environment"
    if scenario != "fresh":
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
        )
        marker.touch()
        site = next((root / "lib").glob("python*/site-packages"))
        (site / "pip.py").touch()
        torch_version = {
            "stale_prebuilt": "2.9.1+cpu",
            "noncpu_prebuilt": "2.10.0",
        }.get(scenario, "2.10.0+cpu")
        (site / "torch.py").write_text(
            "import os\n"
            "assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] == '0'\n"
            f"__version__ = {torch_version!r}\n"
            "class version: cuda = None\n"
        )
        if scenario.startswith("invalid"):
            config = root / "pyvenv.cfg"
            config.write_text(config.read_text().replace("= false", "= true"))

    bootstrap = tmp_path / "bootstrap"
    bootstrap.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        'if [[ "$1" == "-m" && "$2" == "venv" ]]; then\n'
        '  "$REAL_PYTHON" -m venv --without-pip "${@:3}"\n'
        '  "$REAL_PYTHON" - "${!#}" <<\'PY\'\n'
        "import sys\nfrom pathlib import Path\n"
        'site = next((Path(sys.argv[1]) / "lib").glob("python*/site-packages"))\n'
        '(site / "pip.py").touch()\n'
        "PY\n"
        'else\n  exec "$REAL_PYTHON" "$@"\nfi\n'
    )
    bootstrap.chmod(0o755)
    # Fail rather than touching the host if the setup unexpectedly calls apt.
    apt = tmp_path / "apt-get"
    apt.write_text("#!/bin/bash\necho UNEXPECTED_APT >&2\nexit 1\n")
    apt.chmod(0o755)
    start = SCRIPT.index("# --- Isolated Python")
    end = SCRIPT.index("\nexport VIRTUAL_ENV=", start)
    prebuilt = scenario.endswith("prebuilt")
    result = run_shell(
        f'source "{COMMON_PATH}"\n'
        'pip_retry() { printf "PIP %s\\n" "$*"; }\n'
        'install_cpu_torch() { echo "CPU_TORCH_INSTALL $CPU_TORCH_VERSION"; }\n'
        + SCRIPT[start:end]
        + '\necho "AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD"',
        REAL_PYTHON=sys.executable,
        TORCH_FL_BOOTSTRAP_PYTHON=str(bootstrap),
        TORCH_FL_PREBUILT_MUSA_VENV=str(root if prebuilt else tmp_path / "absent"),
        TORCH_FL_VENV_ROOT="" if prebuilt else str(root),
        REPO_ROOT=str(REPO_ROOT),
        CI_STAGE=stage,
        CPU_TORCH_VERSION="2.10.0",
        PIP_INDEX_URL_ARG="https://example.invalid/simple",
        TORCH_FL_INSTALL_QWEN_DEPS="0",
        TORCH_DEVICE_BACKEND_AUTOLOAD="1",
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        PYTHONPATH=str(tmp_path / "inherited-vendor-python-path"),
    )
    if scenario == "invalid_prebuilt":
        assert result.returncode != 0
        assert "Prebuilt MUSA venv is not isolated" in result.stderr
        assert "PIP " not in result.stdout
        assert marker.exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists() == (scenario in ("reused", "prebuilt") or prebuilt)
    assert "--upgrade" not in result.stdout
    assert "ninja build" in result.stdout
    assert ("transformers>=4.51,<5" in result.stdout) == (stage == "integration")
    assert ("CPU_TORCH_INSTALL" in result.stdout) == (
        scenario in ("fresh", "invalid_local", "stale_prebuilt", "noncpu_prebuilt")
    )
    if "CPU_TORCH_INSTALL" in result.stdout:
        assert "CPU_TORCH_INSTALL 2.10.0+cpu" in result.stdout
    assert "AUTOLOAD=1" in result.stdout
