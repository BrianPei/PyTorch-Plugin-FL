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

"""The set_env_*.sh scripts share one helper library.

`pip_retry`, `flag_gems_installed`, `install_flag_gems`, `strip_vendor_paths`
and `venv_is_usable` used to be copied into every platform script -- `pip_retry`
in five slightly different spellings -- so a fix had to be applied seven times
and was easy to miss. They now live in `.github/scripts/lib/set_env_common.sh`,
which every script sources. This keeps it that way: the library defines them,
no script redefines them, and every script sources the library.

Pure text: the scripts cannot be executed here (they provision vendor
environments), but the wiring is fully visible in their source.
"""

import re
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
LIB = SCRIPTS_DIR / "lib" / "set_env_common.sh"
HOOKS_DIR = SCRIPTS_DIR / "hooks"
RUNNER = SCRIPTS_DIR / "set_env.sh"

SHARED_FUNCS = [
    "pip_retry",
    "flag_gems_installed",
    "install_flag_gems",
    "strip_vendor_paths",
    "venv_is_usable",
]

#: The provisioning phases that were byte-for-byte the same modulo the pins.
SHARED_PHASES = [
    "install_cpu_torch",
    "install_flagtree",
    "install_flagcx",
    "build_flagos_inplace",
    "export_ci_env",
]

SOURCE_LINE = 'source "${REPO_ROOT}/.github/scripts/lib/set_env_common.sh"'


@pytest.mark.parametrize(
    "selection,mirror_ok,primary_ok,expected,code",
    [
        ("mirror", "1", "1", ["mirror"], 0),
        ("mirror", "0", "1", ["mirror", "primary"], 0),
        ("mirror", "0", "0", ["mirror", "primary"], 1),
        ("primary", "0", "0", ["primary"], 1),
        ("disabled", "0", "1", ["primary"], 0),
    ],
)
def test_cpu_torch_mirror_falls_back_without_changing_the_pin(
    tmp_path, selection, mirror_ok, primary_ok, expected, code
):
    record = tmp_path / "calls"
    script = r"""
set -euo pipefail
source "$1"
CPU_TORCH_INDEX_URL=https://primary.invalid/cpu
CPU_TORCH_MIRROR_INDEX_URL_DEFAULT=https://mirror.invalid/cpu
CPU_TORCH_VERSION=2.10.0
if [[ "$SELECTION" == disabled ]]; then
  export TORCH_FL_CPU_TORCH_MIRROR_INDEX_URL=''
else
  unset TORCH_FL_CPU_TORCH_MIRROR_INDEX_URL
fi
faster_cpu_torch_index() {
  if [[ "$SELECTION" == primary ]]; then printf '%s' "$1"; else printf '%s' "$2"; fi
}
pip_retry() {
  printf '%s\n' "$*" >> "$RECORD"
  if [[ "$2" == https://mirror.invalid/cpu ]]; then
    [[ "$MIRROR_OK" == 1 ]]
  else
    [[ "$PRIMARY_OK" == 1 ]]
  fi
}
install_cpu_torch --only-binary=:all:
"""
    result = subprocess.run(
        ["bash", "-c", script, "mirror-test", str(LIB)],
        env={
            **os.environ,
            "SELECTION": selection,
            "MIRROR_OK": mirror_ok,
            "PRIMARY_OK": primary_ok,
            "RECORD": str(record),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stdout + result.stderr
    assert record.read_text().splitlines() == [
        f"--index-url https://{source}.invalid/cpu --only-binary=:all: torch==2.10.0"
        for source in expected
    ]


def _scripts() -> list[Path]:
    """The per-platform hook bodies (the old script bodies)."""
    return sorted(HOOKS_DIR.glob("set_env_*.sh"))


def test_the_library_defines_every_shared_function():
    text = LIB.read_text(encoding="utf-8")
    for fn in SHARED_FUNCS:
        assert re.search(rf"^{fn}\(\) \{{", text, re.M), fn


def test_no_script_redefines_a_shared_function():
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        for fn in SHARED_FUNCS:
            assert not re.search(rf"^{fn}\(\) \{{", text, re.M), f"{script.name}: {fn}"


def test_the_runner_sources_the_library():
    """One entrypoint sources the library; the hooks inherit it."""
    assert SOURCE_LINE in RUNNER.read_text(encoding="utf-8")


def test_the_cuda_script_still_selects_its_interpreter_per_call():
    """CUDA drives several interpreters, so it must set PIP_RETRY_PYTHON.

    The shared pip_retry defaults to $VENV_PYTHON; CUDA's bootstrap/vendor steps
    need a different one, so every call has to name it or the wrong interpreter
    is used silently.
    """
    text = (HOOKS_DIR / "set_env_nvidia.sh").read_text(encoding="utf-8")
    calls = re.findall(r"pip_retry ", text)
    prefixed = re.findall(r'PIP_RETRY_PYTHON="\$[A-Za-z_]+" pip_retry ', text)
    assert calls and len(calls) == len(prefixed), (len(calls), len(prefixed))


def test_the_library_defines_every_shared_phase():
    text = LIB.read_text(encoding="utf-8")
    for fn in SHARED_PHASES:
        assert re.search(rf"^{fn}\(\) \{{", text, re.M), fn


def test_no_script_redefines_a_shared_phase():
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        for fn in SHARED_PHASES:
            assert not re.search(rf"^{fn}\(\) \{{", text, re.M), f"{script.name}: {fn}"


def test_every_script_calls_the_common_phases():
    """Shared phases remain shared; MPS and Iluvatar have special provisioning.

    install_flagcx is defined for the platforms whose hook installs a published
    FlagCX wheel, which is not every one of them, so it is not asserted here.
    FlagTree is absent on MPS, which has no Metal Triton build. Iluvatar keeps
    CPU torch in its image but can replace the compiler cohort through a helper
    that delegates installation to the common phases. build_flagos_inplace runs
    only at the build stage, however, so only a hook that calls it there can
    avoid compiling an extension during integration.
    """
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        assert re.search(r"^[ \t]*export_ci_env\b", text, re.M), (
            f"{script.name}: export_ci_env"
        )
        if script.name == "set_env_apple.sh":
            assert not re.search(r"^[ \t]*install_flagtree\b", text, re.M)
            assert re.search(
                r'^if \[\[ "\$CI_STAGE" == "build" \]\]; then\n'
                r"  # [^\n]+\n\s+"
                r"build_flagos_inplace\nfi$",
                text,
                re.M,
            ), "set_env_apple.sh: build_flagos_inplace must be gated on build"
        elif script.name == "set_env_iluvatar.sh":
            assert not re.search(r"^[ \t]*install_cpu_torch\b", text, re.M)
            for phase in ("install_flagtree", "build_flagos_inplace"):
                phase_text = (
                    (SCRIPTS_DIR / "lib/iluvatar_packages.sh").read_text()
                    if phase == "install_flagtree"
                    else text
                )
                assert re.search(rf"^[ \t]*{phase}\b", phase_text, re.M), (
                    f"{script.name}: {phase}"
                )
        else:
            for phase in (
                "install_cpu_torch",
                "install_flagtree",
                "build_flagos_inplace",
            ):
                assert re.search(rf"^[ \t]*{phase}\b", text, re.M), (
                    f"{script.name}: {phase}"
                )


def test_every_local_in_the_library_is_initialized():
    """Every `local` in the library names a value, not a slot.

    The scripts run under `set -u` (`.github/scripts/set_env.sh`), and bash 5 --
    what the runners use -- reports a bare `local` that has not been assigned
    yet as an unbound variable the moment it is expanded. `local a b="" c` is
    therefore safe for `b` and `c` and a landmine for `a`: the first PR to add
    `pip_index_host` declared `local arg prev="" url host` and assigned `url`
    inside a `case` branch, so the Ascend build job died with
    `line 54: url: unbound variable` (run 38491036055) while the same code
    passed on macOS, whose bash 3.2 expands an unassigned local to the empty
    string instead. A name the `for` statement assigns every iteration is the
    one shape a bare declaration may have.
    """
    text = LIB.read_text(encoding="utf-8")
    functions = re.split(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(\) \{", text, flags=re.M)
    assert functions, "no functions parsed"
    for name, body in zip(functions[1::2], functions[2::2], strict=True):
        # A function body ends at the first closing brace in column 0.
        body = body.split("\n}", 1)[0]
        for declaration in re.finditer(r"^[ \t]*local ([^\n]*)$", body, re.M):
            for word in declaration.group(1).split():
                variable = word.split("=", 1)[0]
                if "=" in word or not variable.isidentifier():
                    continue
                assert re.search(rf"\bfor\s+{variable}\s+in\b", body), (
                    f"{name}: `local {variable}` has no initializer and is not a "
                    f"for-loop variable; under `set -u` bash 5 fails on it"
                )
