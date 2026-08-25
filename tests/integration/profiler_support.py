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

"""Shared fixtures and helpers for the cross-backend profiler contract.

Three layers are kept separate so a backend that cannot yet satisfy the full
public contract still leaves evidence instead of being whitelisted out:

1. Environment preflight -- the integration conftest exits early when no flagos
   device is available (``pytest.exit`` in ``conftest.py``). That decides whether
   the backend is present at all, not how much profiling it can do.
2. Required Contract -- ``test_profiler_contract.py`` asserts a full-featured
   profiler (kernel, runtime, memcpy, memset, flow, linkage, metadata). There is
   no per-platform capability table and no skip: if a backend emits none of a
   category, the test fails.
3. Observed result -- ``submit_observation`` records what the profiler actually
   emitted on this box, and ``pytest_sessionfinish`` writes ``profiler-observations.json``
   under ``REPORT_DIR`` (``integration-reports/``). This is report-only and never
   gates a test, so a genuinely missing capability is visible in CI rather than
   silently dropped.
"""

import inspect
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from platform_support import detect_platform


@dataclass(frozen=True)
class ProfilerCapabilities:
    """The Required Contract a full-featured profiler must satisfy.

    ``platform`` labels the active backend. The feature flags are constants,
    never switches: every supported backend is expected to emit all of them.
    Kept as a dataclass so the contract reads as one object; no test may use a
    flag to skip.
    """

    platform: str
    device: bool = True
    kernel: bool = True
    runtime: bool = True
    memcpy: bool = True
    memset: bool = True
    flow: bool = True
    linkage: bool = True
    metadata: bool = True


def _torch_device():
    import torch

    return torch.device("flagos", 0)


def _torch_module():
    import torch

    return torch


def required_capabilities() -> ProfilerCapabilities:
    """The Required Contract for the active backend (report label only)."""
    return ProfilerCapabilities(platform=detect_platform())


@pytest.fixture(scope="session")
def profiler_capabilities():
    """Required Contract capabilities for the active backend."""
    return required_capabilities()


# ---------------------------------------------------------------------------
# Layer 3: observed result. Purely report-only; nothing here may skip a test.
# ---------------------------------------------------------------------------

_OBSERVATION_STORE: dict = {}


def submit_observation(module: str, trace) -> None:
    """Record what the active tracer actually emitted for ``module``.

    ``module`` is the test module that produced ``trace`` (captured by the
    ``profile_result`` fixture). Aggregated at session finish and written by
    ``publish_observations``.
    """
    events = trace.get("traceEvents", [])
    _OBSERVATION_STORE[module] = {
        "categories": sorted(event_categories(trace)),
        "total_events": len(events),
        "kernel": len(events_in(trace, "kernel")),
        "gpu_memcpy": len(events_in(trace, "gpu_memcpy")),
        "gpu_memset": len(events_in(trace, "gpu_memset")),
        "privateuse1_runtime": len(events_in(trace, "privateuse1_runtime")),
        "flow": len(
            [e for e in events if e.get("cat") == "ac2g" and e.get("ph") in {"s", "f"}]
        ),
    }


def _observation_path() -> Path:
    """Where to write profiler-observations.json.

    CI sets ``REPORT_DIR`` (see the integration runner). A local run falls back
    to ``integration-reports/`` under the repository root.
    """
    report_dir = os.environ.get("REPORT_DIR")
    if report_dir:
        return Path(report_dir) / "profiler-observations.json"
    return (
        Path(__file__).resolve().parents[2]
        / "integration-reports"
        / "profiler-observations.json"
    )


def publish_observations(exitstatus: int) -> None:
    """Write the observed result, even when the contract failed."""
    if not _OBSERVATION_STORE:
        return
    record = {
        "platform": detect_platform(),
        "pytest_exitstatus": exitstatus,
        "observed_modules": [
            {"module": module, "metrics": metrics}
            for module, metrics in sorted(_OBSERVATION_STORE.items())
        ],
    }
    path = _observation_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[profiler] wrote observed result: {path}")


def pytest_sessionfinish(session, exitstatus):
    """Pytest hook: publish the observed profiler result at session end."""
    # Kept above the fixture so a trace failure still writes the report.
    publish_observations(exitstatus)


def _consumer_module() -> str:
    """Name of the test module that requested the ``profile_result`` fixture.

    The fixture is solved while a test module is being set up, so the module
    whose tests requested it is on the stack. ``profile_result`` itself lives
    here in ``profiler_support``, so the walk skips this module and returns the
    first other ``tests.integration`` module.
    """
    frame = inspect.currentframe()
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module.startswith("tests.integration") and module != __name__:
            return module
        frame = frame.f_back
    return "unknown-module"


@pytest.fixture(scope="module")
def profile_result():
    """Capture one common workload and export it as a Chrome trace.

    Shape and iteration count are kept identical to ``_run_traced_ops()`` in
    test_profiler_parity.py, which is the workload proven to emit every activity
    class this module asserts on. It matters for memsets specifically: cuBLAS
    only allocates (and zeroes) a gemm workspace once the matmul is large enough
    and repeated enough to pick a workspace-using kernel. A 256x256 x3 loop stays
    under that threshold on CUDA and produced no gpu_memset events at all, while
    still passing on backends whose sort allocates zeroed scratch -- so shrinking
    this workload silently converts the memset assertion into a no-op on some
    vendors and a failure on others.
    """
    torch = _torch_module()
    device = _torch_device()
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    small = torch.randn(16, device=device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        with_stack=False,
    ) as prof:
        for _ in range(5):
            z = (x @ y).relu()

        # Trigger memset: zero initialization
        _ = torch.zeros(128, 128, device=device)

        # Trigger memcpy: host-to-device and device-to-host transfers
        cpu_tensor = torch.randn(64, 64)
        device_copy = cpu_tensor.to(device)
        _ = device_copy.cpu()

        torch.sort(small)
        z.sum().item()  # force sync so device activity lands inside the window

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as trace_file:
        trace_path = Path(trace_file.name)
    try:
        prof.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
    finally:
        trace_path.unlink(missing_ok=True)
    submit_observation(_consumer_module(), trace)
    return prof, trace


def events_in(trace, category, *, completed_only=True):
    """Return trace events in one category."""
    events = [
        event for event in trace.get("traceEvents", []) if event.get("cat") == category
    ]
    if completed_only:
        return [event for event in events if event.get("ph") == "X"]
    return events


def event_categories(trace):
    """Return categories for completed trace events."""
    return {
        event.get("cat")
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X"
    }


def arg_key_union(trace, category):
    """Return the union of argument keys for all events in a category."""
    keys = set()
    for event in events_in(trace, category):
        keys.update((event.get("args") or {}).keys())
    return keys


def op_name_by_external_id(trace):
    """Map Kineto External ids to CPU operation names."""
    return {
        (event.get("args") or {}).get("External id"): event.get("name")
        for event in events_in(trace, "cpu_op")
        if (event.get("args") or {}).get("External id") is not None
    }
