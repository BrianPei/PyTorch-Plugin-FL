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

"""Compatibility checks against the installed candidate, outside the checkout."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def test_installed_candidate_and_dependencies():
    import torch_fl
    import torch

    package = Path(torch_fl.__file__).resolve()
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    assert workspace not in package.parents, (
        f"source tree shadowed installed wheel: {package}"
    )
    record = package.parent / "compatibility.json"
    assert record.is_file(), "installed wheel has no compatibility record"
    assert json.loads(record.read_text())
    assert importlib.metadata.version("torch_fl")
    assert torch.__version__
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, timeout=120)
    args = [
        "torch-fl-preflight",
        "--installed",
        "--check-installed",
        "--check-build-env",
        "--strict-tested",
    ]
    if os.environ["FLAGCICD_PROFILE"] == "release":
        args += ["--release", "--require-sdk"]
    subprocess.run(args, check=True, timeout=120)


def test_fresh_process_loads_native_library():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch_fl; assert torch_fl.flagos.is_available(); torch_fl.flagos.init()",
        ],
        check=True,
        timeout=120,
    )
