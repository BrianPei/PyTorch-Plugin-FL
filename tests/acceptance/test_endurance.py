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

"""Fixed workload stability; duration is recorded and never shortened on release."""

import json
import os
from pathlib import Path
import time

from test_lifecycle import cycle


def test_candidate_endurance():
    import torch_fl

    duration = int(os.environ["FLAGCICD_STABILITY_SECONDS"])
    minimum = 3600 if os.environ["FLAGCICD_PROFILE"] == "release" else 600
    assert duration >= minimum
    cycle()
    initial = torch_fl.flagos.memory_allocated()
    started = time.monotonic()
    iterations = 0
    while time.monotonic() - started < duration:
        cycle()
        iterations += 1
        assert torch_fl.flagos.memory_allocated() - initial <= int(
            os.environ["FLAGCICD_MAX_RETAINED_BYTES"]
        )
    metrics = {
        "iterations": iterations,
        "duration_seconds": time.monotonic() - started,
        "workload": "flagos-fp32-128x128-square-backward-stream-free/v1",
    }
    Path(os.environ["FLAGCICD_METRICS_DIR"]).joinpath("endurance.json").write_text(
        json.dumps(metrics)
    )
