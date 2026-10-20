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

"""Release model acceptance reuses the existing installed-wheel model suites."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("suite", ["test_qwen3_infer.py", "test_qwen3_train.py"])
def test_candidate_model(suite):
    model = Path(os.environ["MODEL_PATH"])
    assert model.is_dir(), f"missing mandatory acceptance model: {model}"
    # Keep native child pytest reporting enabled: an outer passing wrapper alone
    # must not hide empty collection or an all-skipped model invocation.
    env = os.environ.copy()
    env["FLAGCICD_REPORT_DIR"] = env["FLAGCICD_METRICS_DIR"]
    env["FLAGCICD_TEST_GROUP"] = "l4_models"
    env["PYTEST_PLUGINS"] = "flagcicd_report"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"tests/integration/{suite}",
            "--model",
            str(model),
            "-v",
            "--tb=short",
        ],
        check=True,
        env=env,
        timeout=3300,
    )
