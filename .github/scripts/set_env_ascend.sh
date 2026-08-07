#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ascend NPU environment for the common build/integration workflow.
#
# Ascend is a standalone vendor backend (pure CANN ACLNN). Unlike CUDA it has no
# boxing layer, and unlike MetaX it does not shim a CUDA runtime. The wheel is
# built against a stock CPU PyTorch (2.10.0+cpu); the ACLNN shared libraries
# from the CANN toolkit are linked at runtime via LD_LIBRARY_PATH.
set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- CANN toolkit root -------------------------------------------------------
# Both layouts seen in the field:
#   /usr/local/Ascend/ascend-toolkit/latest   (symlink into the platform dir)
#   /usr/local/Ascend/cann-<ver>/<arch>-linux
# ASCEND_HOME must expose $ASCEND_HOME/include/aclnnop and $ASCEND_HOME/lib64.
ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ ! -d "$ASCEND_HOME/lib64" || ! -d "$ASCEND_HOME/include" ]]; then
  echo "::error::CANN toolkit not found at $ASCEND_HOME (need lib64/ and include/)"
  exit 1
fi

# Runtime ACLNN libraries actually linked by libtorch_fl.so. Their presence is
# the minimum proof that the image can drive an Ascend kernel.
for lib in libascendcl.so libopapi.so libnnopbase.so; do
  if ! compgen -G "$ASCEND_HOME/lib64/$lib*" >/dev/null \
     && ! compgen -G "$ASCEND_HOME/acllib/lib64/$lib*" >/dev/null; then
    echo "::error::Required ACLNN library missing: $lib (looked in $ASCEND_HOME/lib64 and $ASCEND_HOME/acllib/lib64)"
    exit 1
  fi
done

# --- Device node -------------------------------------------------------------
# Ascend exposes a manager device plus per-card davinci nodes. Require the
# manager node; card count is asserted later from torch_fl.flagos.device_count().
if [[ ! -c /dev/davinci_manager ]]; then
  echo "::error::Ascend device node /dev/davinci_manager is unavailable"
  exit 1
fi

# --- Environment -------------------------------------------------------------
export ACCELERATOR=ascend
export ASCEND_HOME
# Ascend has no CUDA assets, no CUDA runtime, and (in the first-version wheel)
# no FlagGems C++/Python path. Disable all of them explicitly so dispatch
# resolves every op to the ascend ACLNN backend.
export FLAGOS_DISABLE_CUDA_ASSETS=1
export FLAGOS_USE_FLAGGEMS=0
export FLAGOS_USE_FLAGGEMS_CPP=0
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON=0
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

# ACLNN headers for the build (aclnnop/*.h, acl/*.h).
export CPATH="$ASCEND_HOME/include${CPATH:+:$CPATH}"
# Link-time and runtime library path. lib64 holds libascendcl/libopapi/libnnopbase;
# acllib/lib64 is the legacy fallback on some CANN images.
export LIBRARY_PATH="$ASCEND_HOME/lib64:$ASCEND_HOME/acllib/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ASCEND_HOME/lib64:$ASCEND_HOME/acllib/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Build Python / CPU torch ------------------------------------------------
# The vendor Python in the CANN image already has a CPU torch (2.10.0+cpu). We
# reuse it directly: ascend needs no accelerator-linked torch, only the CPU
# dispatcher plus the ACLNN runtime libraries above. A fresh venv isolates the
# install from whatever else the image carries.
VENDOR_PYTHON="${TORCH_FL_VENDOR_PYTHON:-$(command -v python)}"
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  echo "::error::Unable to find the build Python interpreter"
  exit 1
fi
if ! "$VENDOR_PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "::error::Build Python cannot import torch: $VENDOR_PYTHON" >&2
  exit 1
fi

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-ascend-${CI_STAGE}}"
if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
  echo "::warning::Build Python cannot create a venv; trying uv"
  if ! command -v uv >/dev/null 2>&1; then
    "$VENDOR_PYTHON" -m pip install --upgrade uv
  fi
  uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel cmake
"$VENV_PYTHON" -m pip install --index-url "${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
  "torch==${TORCH_FL_CPU_TORCH_VERSION:-2.10.0}"
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" -m pip install pytest transformers
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# Sanity: the isolated torch must be the CPU wheel, not a CUDA vendor build.
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR ASCEND_HOME \
    FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_USE_FLAGGEMS FLAGOS_USE_FLAGGEMS_CPP \
    FLAGGEMS_KERNEL FLAGGEMS_PYTHON CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

cd "$REPO_ROOT"

if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so before the common workflow
  # invokes python -m build. The following wheel build is incremental.
  python setup.py build_ext --inplace
fi

# Final availability check: the flagos device must come up on top of the CPU
# torch and the ACLNN runtime. This mirrors the integration health check.
python - <<'PY'
import torch_fl
import torch

assert torch_fl.flagos.is_available(), "flagos device is unavailable"
n = torch_fl.flagos.device_count()
assert n >= 1, f"expected >=1 flagos device, got {n}"
print(f"flagos devices: {n}")
PY
