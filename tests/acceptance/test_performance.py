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

"""Fixed workload performance evidence; an absent baseline cannot pass release."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def test_fixed_workload_performance():
    import torch_fl
    import torch

    runtime = torch_fl.flagos
    x = torch.ones((512, 512), device="flagos:0")

    def workload():
        result = (x * x + x).sum()
        runtime.synchronize()
        assert result.cpu().item() == 524288.0

    for _ in range(5):
        workload()
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        workload()
        samples.append(time.perf_counter() - start)
    compatibility = json.loads(
        (Path(torch_fl.__file__).parent / "compatibility.json").read_text()
    )
    identity = {
        "build": compatibility["build"],
        "device": runtime.get_device_properties(0).name,
        "python": __import__("platform").python_version(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    baselines = json.loads(
        (
            Path(os.environ["GITHUB_WORKSPACE"])
            / ".flagcicd/performance-baselines.json"
        ).read_text()
    )
    baseline = baselines["environments"].get(key)
    metrics = {
        "workload": "fp32-512x512-mul-add-sum-copy/v1",
        "environment": identity,
        "baseline_key": key,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "baseline_status": "configured" if baseline else "uncalibrated",
    }
    Path(os.environ["FLAGCICD_METRICS_DIR"]).joinpath("performance.json").write_text(
        json.dumps(metrics, indent=2)
    )
    if baseline:
        assert baseline["workload"] == metrics["workload"]
        assert (
            baseline["sample_count"] >= 20
            and baseline["approved_by"]
            and baseline["reference_run"]
        )
        assert 0 < metrics["median_seconds"] <= baseline["max_median_seconds"]
    elif os.environ["FLAGCICD_PROFILE"] == "release":
        raise AssertionError(
            f"Release requires an approved performance baseline for {key}"
        )
