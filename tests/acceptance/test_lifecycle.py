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

"""Exercise allocation, streams, backward, cleanup and recovery on flagos."""

import gc
import os


def cycle():
    import torch
    import torch_fl

    runtime = torch_fl.flagos
    stream = runtime.Stream()
    with runtime.stream(stream):
        x = torch.ones((128, 128), device="flagos:0", requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
    stream.synchronize()
    assert torch.isfinite(loss).item()
    torch.testing.assert_close(x.grad.cpu(), torch.full((128, 128), 2.0))
    del x, loss, stream
    gc.collect()
    runtime.synchronize()
    runtime.empty_cache()


def test_repeated_create_run_free_and_recovery():
    import torch_fl

    runtime = torch_fl.flagos
    cycle()  # Warm lazy initialization before measuring retained allocations.
    before = runtime.memory_allocated()
    for _ in range(20):
        cycle()
    after = runtime.memory_allocated()
    assert after - before <= int(
        os.environ.get("FLAGCICD_MAX_RETAINED_BYTES", "16777216")
    )
    import torch

    interrupted = torch.ones((128, 128), device="flagos:0")
    try:
        raise RuntimeError("intentional host-side workload interruption")
    except RuntimeError:
        del interrupted
        gc.collect()
        runtime.empty_cache()
    cycle()
