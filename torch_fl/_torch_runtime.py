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

"""Load accelerator-side PyTorch libraries bundled with the torch_fl wheel."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path


_loaded_handles: list[ctypes.CDLL] = []


def load_bundled_torch_runtime() -> list[ctypes.CDLL]:
    """Preload the staged vendor libraries with global symbol visibility.

    The common PyTorch libraries have already been loaded from the upstream CPU
    wheel before this function is called. Only accelerator-side libraries live
    in ``torch_fl/lib/torch_runtime``.
    """

    if _loaded_handles:
        return _loaded_handles
    if os.environ.get("FLAGOS_DISABLE_BUNDLED_TORCH_RUNTIME", "0") == "1":
        return _loaded_handles

    runtime_dir = Path(__file__).resolve().parent / "lib" / "torch_runtime"
    manifest_path = runtime_dir / "manifest.json"
    if not manifest_path.is_file():
        return _loaded_handles

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError(
            f"unsupported bundled torch runtime manifest: {manifest_path}"
        )

    load_order = [
        *manifest.get("preload", []),
        *manifest.get("load_order", []),
    ]
    for library in load_order:
        if Path(library).name != library:
            raise RuntimeError(f"invalid library name in {manifest_path}: {library!r}")
        library_path = runtime_dir / library
        if not library_path.is_file():
            raise RuntimeError(
                f"bundled torch runtime library is missing: {library_path}"
            )
        try:
            handle = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise RuntimeError(
                f"failed to load bundled torch runtime library {library_path}: {error}"
            ) from error
        _loaded_handles.append(handle)

    return _loaded_handles
