#!/usr/bin/env python3
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

"""Build and register the MetaX libcudart compatibility shim in the wheel."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("csrc/runtime/accelerator/metax"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("torch_fl/lib/torch_runtime"),
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    destination = args.destination.resolve()
    shim_source = source_dir / "cudart_shim.c"
    version_script = source_dir / "libcudart.version"
    manifest_path = destination / "manifest.json"
    output = destination / "libcudart.so.12"

    for required in (shim_source, version_script, manifest_path):
        if not required.is_file():
            raise RuntimeError(f"Required MetaX runtime input is missing: {required}")

    subprocess.check_call(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-o",
            str(output),
            str(shim_source),
            f"-Wl,--version-script={version_script}",
            "-Wl,-soname,libcudart.so.12",
            "-ldl",
        ]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preload = [
        name for name in manifest.get("preload", []) if name != output.name
    ]
    manifest["preload"] = [output.name, *preload]
    libraries = set(manifest.get("libraries", []))
    libraries.add(output.name)
    manifest["libraries"] = sorted(libraries)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Staged MetaX cudart shim: {output}")


if __name__ == "__main__":
    main()
