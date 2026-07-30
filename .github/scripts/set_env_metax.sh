#!/usr/bin/env bash
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

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-2.10.0}"
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-metax-${CI_STAGE}-py3.12}"
RUNTIME_DIR="$REPO_ROOT/torch_fl/lib/torch_runtime"

select_vendor_python() {
  local candidate
  if [[ -n "${TORCH_FL_VENDOR_PYTHON:-}" ]]; then
    candidate="$TORCH_FL_VENDOR_PYTHON"
    if [[ "$candidate" != */* ]]; then
      candidate=$(command -v "$candidate" 2>/dev/null || true)
    fi
    if [[ -z "$candidate" ]]; then
      echo "::error::TORCH_FL_VENDOR_PYTHON was not found" >&2
      exit 1
    fi
    if ! "$candidate" -c "import torch" >/dev/null 2>&1; then
      echo "::error::TORCH_FL_VENDOR_PYTHON cannot import torch: $candidate" >&2
      exit 1
    fi
    printf '%s' "$candidate"
    return
  fi
  for candidate in python python3 python3.12; do
    candidate=$(command -v "$candidate" 2>/dev/null || true)
    [[ -z "$candidate" ]] && continue
    if "$candidate" -c "import torch" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return
    fi
  done
  echo "::error::Unable to find a Python interpreter with the vendor torch package" >&2
  exit 1
}

VENDOR_PYTHON=$(select_vendor_python)
if [[ ! -x "$VENDOR_PYTHON" ]]; then
  echo "::error::Vendor Python is unavailable: $VENDOR_PYTHON"
  exit 1
fi

VENDOR_INFO=$("$VENDOR_PYTHON" - <<'PY'
import json
import sys
from pathlib import Path

import torch

print(
    json.dumps(
        {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "version": torch.__version__,
            "base_version": torch.__version__.split("+", 1)[0],
            "cuda": torch.version.cuda,
            "git_version": getattr(torch.version, "git_version", None),
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "root": str(Path(torch.__file__).resolve().parent),
        },
        sort_keys=True,
    )
)
PY
)

readarray -t VENDOR_FIELDS < <(
  VENDOR_INFO="$VENDOR_INFO" "$VENDOR_PYTHON" - <<'PY'
import json
import os

info = json.loads(os.environ["VENDOR_INFO"])
for key in (
    "python",
    "version",
    "base_version",
    "cuda",
    "git_version",
    "cxx11_abi",
    "root",
):
    value = info.get(key)
    print("" if value is None else value)
PY
)
VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[0]}"
VENDOR_TORCH_VERSION="${VENDOR_FIELDS[1]}"
VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[2]}"
VENDOR_CUDA_VERSION="${VENDOR_FIELDS[3]}"
VENDOR_TORCH_GIT_VERSION="${VENDOR_FIELDS[4]}"
VENDOR_CXX11_ABI="${VENDOR_FIELDS[5]}"
VENDOR_TORCH_ROOT="${VENDOR_FIELDS[6]}"
VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

if [[ "$VENDOR_PYTHON_VERSION" != "3.12" ]]; then
  echo "::error::MetaX vendor image must provide Python 3.12, got $VENDOR_PYTHON_VERSION"
  exit 1
fi
if [[ "$VENDOR_TORCH_BASE_VERSION" != "$CPU_TORCH_VERSION" ]]; then
  echo "::error::Vendor torch is $VENDOR_TORCH_VERSION; expected base version $CPU_TORCH_VERSION"
  exit 1
fi

echo "Vendor PyTorch: $VENDOR_TORCH_VERSION"
echo "Vendor PyTorch CUDA ABI: ${VENDOR_CUDA_VERSION:-unknown}"
echo "Vendor PyTorch root: $VENDOR_TORCH_ROOT"
echo "Vendor PyTorch git: ${VENDOR_TORCH_GIT_VERSION:-unknown}"
echo "Vendor PyTorch CXX11 ABI: $VENDOR_CXX11_ABI"

"$VENDOR_PYTHON" "$REPO_ROOT/.github/scripts/stage_torch_accelerator_libs.py" \
  --source "$VENDOR_TORCH_LIB" \
  --destination "$RUNTIME_DIR"
"$VENDOR_PYTHON" "$REPO_ROOT/.github/scripts/stage_metax_cudart_shim.py" \
  --source-dir "$REPO_ROOT/csrc/runtime/accelerator/metax" \
  --destination "$RUNTIME_DIR"

detect_metax_path() {
  local candidate
  for candidate in \
    "${METAX_PATH:-}" \
    "${METAX_HOME:-}" \
    "${MACA_PATH:-}" \
    "${MACA_HOME:-}" \
    /opt/maca \
    /opt/maca-* \
    /usr/local/maca; do
    [[ -z "$candidate" ]] && continue
    if [[ -x "$candidate/tools/cu-bridge/bin/cucc" ]]; then
      printf '%s' "$candidate"
      return
    fi
  done
  return 1
}

METAX_PATH=$(detect_metax_path || true)
if [[ -z "$METAX_PATH" ]]; then
  echo "::error::MetaX SDK with tools/cu-bridge/bin/cucc was not found"
  exit 1
fi
export METAX_PATH
export MACA_PATH="$METAX_PATH"
export MACA_HOME="$METAX_PATH"

"$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"
VENV_PYTHON="$VENV_ROOT/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install \
  --index-url "$CPU_TORCH_INDEX_URL" \
  "torch==$CPU_TORCH_VERSION"
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" -m pip install pytest
fi

CPU_INFO=$("$VENV_PYTHON" - <<'PY'
import json
from pathlib import Path

import torch

print(
    json.dumps(
        {
            "version": torch.__version__,
            "base_version": torch.__version__.split("+", 1)[0],
            "cuda": torch.version.cuda,
            "git_version": getattr(torch.version, "git_version", None),
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "root": str(Path(torch.__file__).resolve().parent),
        },
        sort_keys=True,
    )
)
PY
)

VENDOR_INFO="$VENDOR_INFO" CPU_INFO="$CPU_INFO" "$VENV_PYTHON" - <<'PY'
import json
import os

vendor = json.loads(os.environ["VENDOR_INFO"])
cpu = json.loads(os.environ["CPU_INFO"])

assert cpu["base_version"] == vendor["base_version"], (cpu, vendor)
assert cpu["cuda"] is None, (
    "The isolated build environment must use the CPU torch wheel, "
    f"got torch.version.cuda={cpu['cuda']!r}"
)
assert cpu["cxx11_abi"] == vendor["cxx11_abi"], (
    "CPU and vendor torch CXX11 ABI settings differ",
    cpu,
    vendor,
)
assert cpu["root"] != vendor["root"], (
    "The isolated environment still imports the vendor torch package",
    cpu,
    vendor,
)

print(f"CPU PyTorch: {cpu['version']}")
print(f"CPU PyTorch root: {cpu['root']}")
print(f"CPU PyTorch git: {cpu['git_version'] or 'unknown'}")
if cpu["git_version"] != vendor["git_version"]:
    print(
        "::warning::CPU and vendor torch git revisions differ; "
        "the hardware job must validate the native ABI"
    )
PY

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$METAX_PATH/tools/cu-bridge/bin:$METAX_PATH/bin:$METAX_PATH/mxgpu_llvm/bin:$PATH"
export ACCELERATOR=metax
export TORCH_FL_VENDOR_TORCH_ROOT="$VENDOR_TORCH_ROOT"
export TORCH_FL_BUNDLED_TORCH_RUNTIME="$RUNTIME_DIR"
export TORCH_FL_REQUIRE_ISOLATED_TORCH=1
export TORCH_FL_CPU_TORCH_VERSION="$CPU_TORCH_VERSION"
export TORCH_FL_REQUIRED_ACCELERATOR_LIBS="libc10_cuda.so,libtorch_cuda.so,libcudart.so.12"
export FLAGOS_METAX_CUDART_SHIM=0
export FLAGOS_METAX_COMPAT=0
export FLAGOS_DISABLE_FLAGGEMS_PY=1
export PYTHONPATH=""

CPU_TORCH_ROOT=$("$VENV_PYTHON" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
PY
)

strip_vendor_python_package_paths() {
  local value="${1:-}"
  local entry
  local -a entries=()
  local -a kept=()
  IFS=: read -ra entries <<< "$value"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      "$VENDOR_TORCH_ROOT"|"$VENDOR_TORCH_ROOT"/*|*/site-packages/torch|*/site-packages/torch/*|*/site-packages/flag_gems|*/site-packages/flag_gems/*) ;;
      *) kept+=("$entry") ;;
    esac
  done
  local joined=""
  for entry in "${kept[@]}"; do
    joined="${joined:+$joined:}$entry"
  done
  printf '%s' "$joined"
}

CLEAN_CMAKE_PREFIX_PATH=$(strip_vendor_python_package_paths "${CMAKE_PREFIX_PATH:-}")
CLEAN_LIBRARY_PATH=$(strip_vendor_python_package_paths "${LIBRARY_PATH:-}")
CLEAN_LD_LIBRARY_PATH=$(strip_vendor_python_package_paths "${LD_LIBRARY_PATH:-}")

export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$METAX_PATH/tools/cu-bridge/include:$METAX_PATH/include:$METAX_PATH/include/mcr${CPATH:+:$CPATH}"
export LIBRARY_PATH="$METAX_PATH/lib:$METAX_PATH/tools/cu-bridge/lib:$METAX_PATH/mxgpu_llvm/lib${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CPU_TORCH_ROOT/lib:$METAX_PATH/lib:$METAX_PATH/tools/cu-bridge/lib:$METAX_PATH/mxgpu_llvm/lib${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
  printf '%s\n' "$METAX_PATH/tools/cu-bridge/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    VIRTUAL_ENV ACCELERATOR METAX_PATH MACA_PATH MACA_HOME \
    TORCH_FL_VENDOR_TORCH_ROOT TORCH_FL_BUNDLED_TORCH_RUNTIME \
    TORCH_FL_REQUIRE_ISOLATED_TORCH TORCH_FL_CPU_TORCH_VERSION \
    TORCH_FL_REQUIRED_ACCELERATOR_LIBS \
    FLAGOS_METAX_CUDART_SHIM FLAGOS_METAX_COMPAT \
    FLAGOS_DISABLE_FLAGGEMS_PY \
    CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH PYTHONPATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

if [[ ! -e /dev/mxcd ]]; then
  echo "::warning::/dev/mxcd is not visible; device validation may fail"
fi
if command -v mx-smi >/dev/null 2>&1; then
  mx-smi || true
elif command -v maca-smi >/dev/null 2>&1; then
  maca-smi || true
fi

"$VENV_PYTHON" - <<'PY'
import json
from pathlib import Path

import torch

assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda

runtime_dir = Path("torch_fl/lib/torch_runtime")
manifest = json.loads((runtime_dir / "manifest.json").read_text())
assert "libcudart.so.12" in manifest["preload"], manifest
assert list(runtime_dir.glob("libtorch_cuda.so*")), runtime_dir
assert list(runtime_dir.glob("libc10_cuda.so*")), runtime_dir

print(f"Isolated PyTorch: {torch.__version__}")
print(f"Runtime libraries: {len(list(runtime_dir.glob('*.so*')))}")
PY
