# MUSA Installation and Known Issues

## Overview

MUSA (Moore Threads Unified System Architecture) support is provided through
native mudnn kernels. Unlike CUDA-compatible platforms that box existing CUDA
kernels, MUSA builds against stock CPU PyTorch 2.10 with no CUDA runtime
dependencies.

## Installation

```bash
pip install torch-fl-*.whl
```

The wheel bundles `libtorch_fl.so` and links directly to the MUSA toolkit
(mudnn, musa_runtime). No separate CUDA installation is required.

## Known Packaging Issues

### Wheel Metadata and pip check

The current MUSA wheel declares a dependency on `torch`, which transitively
pulls in `nvidia-*` packages via torch's own `Requires-Dist`. These CUDA
dependencies are never used at runtime (the MUSA backend calls mudnn directly),
but they cause `pip check` to report unmet dependencies in CPU-only environments.

**Status**: Tracked as a packaging gap for follow-up work. The CI pipeline
explicitly skips `pip check` for MUSA and documents this exception in the
workflow. Runtime correctness is unaffected.

**Workaround**: Install with `--no-deps` or use a virtual environment that
already has PyTorch installed.

## Profiler Support

MUSA profiling is provided by the MUPTI tracer, which emits device, kernel,
memcpy, and memset events. The profiler contract (`test_profiler_contract.py`)
runs on MUSA without capability hedging: if MUPTI cannot emit a required event
category, the test fails and the observed categories are written to
`profiler-observations.json` as evidence.

## torch.compile Support

The current MUSA image ships CPU PyTorch with no Triton backend. torch.compile
tests are configured to fail loudly (not skip) so the missing stack is visible
in CI. Once the MThreads flagtree backend is provisioned and validated, the
compile step will use it and the failure will be resolved.

## Device Availability Check

The MUSA integration manifest asserts:
- `torch.version.cuda is None` (CPU wheel, no CUDA boxing)
- `/opt/conda/` is not in torch's path (stock CPU install)
- `libtorch_fl.so` is bundled in the wheel
- `libtorch_cuda.so` is **not** present (native MUSA, not CUDA boxing)
- `torch_fl.flagos.device_count() > 0`

If any of these fail, the preflight check aborts the suite.
