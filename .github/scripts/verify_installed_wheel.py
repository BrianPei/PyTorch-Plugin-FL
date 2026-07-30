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

"""Verify that the installed torch_fl wheel is isolated from vendor Python torch."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from importlib.metadata import files, metadata
from pathlib import Path


_RPATH_RE = re.compile(r"\((?:RPATH|RUNPATH)\).*?: \[(?P<value>[^\]]*)\]")


def _dynamic_search_paths(path: Path) -> list[str]:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise RuntimeError("readelf is required for wheel isolation verification")
    output = subprocess.check_output(
        [readelf, "-d", str(path)],
        text=True,
        stderr=subprocess.STDOUT,
    )
    paths: list[str] = []
    for line in output.splitlines():
        match = _RPATH_RE.search(line)
        if match:
            paths.extend(part for part in match.group("value").split(":") if part)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--require-device", action="store_true")
    args = parser.parse_args()

    import torch
    import torch_fl

    torch_root = Path(torch.__file__).resolve().parent
    package_root = Path(torch_fl.__file__).resolve().parent
    vendor_root_value = os.environ.get("TORCH_FL_VENDOR_TORCH_ROOT")
    vendor_root = Path(vendor_root_value).resolve() if vendor_root_value else None

    require_isolation = os.environ.get("TORCH_FL_REQUIRE_ISOLATED_TORCH") == "1"
    expected_torch_version = os.environ.get(
        "TORCH_FL_CPU_TORCH_VERSION", "2.10.0"
    )
    if require_isolation:
        assert (
            torch.__version__.split("+", 1)[0] == expected_torch_version
        ), torch.__version__
        assert torch.version.cuda is None, (
            "The installed wheel must run with upstream CPU torch; "
            f"torch.version.cuda={torch.version.cuda!r}"
        )
        if vendor_root is not None:
            assert torch_root != vendor_root, (
                "torch still resolves to the vendor Python package",
                torch_root,
                vendor_root,
            )

    if args.workspace is not None:
        workspace = args.workspace.resolve()
        try:
            package_root.relative_to(workspace)
        except ValueError:
            pass
        else:
            raise AssertionError(
                (
                    "torch_fl was imported from the checkout instead of the "
                    "installed wheel"
                ),
                package_root,
                workspace,
            )

    libraries: list[str] = []
    external_dependencies: list[str] = []
    if require_isolation:
        runtime_dir = package_root / "lib" / "torch_runtime"
        manifest_path = runtime_dir / "manifest.json"
        assert manifest_path.is_file(), manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        libraries = manifest.get("libraries", [])
        external_dependencies = manifest.get("external_dependencies", [])
        required_libraries = [
            name
            for name in os.environ.get(
                "TORCH_FL_REQUIRED_ACCELERATOR_LIBS", ""
            ).split(",")
            if name
        ]
        for required_library in required_libraries:
            assert any(
                name.startswith(required_library) for name in libraries
            ), (required_library, libraries)

        package_files = [str(path) for path in files("torch_fl")]
        for required_library in required_libraries:
            assert any(
                f"torch_runtime/{required_library}" in path for path in package_files
            ), (required_library, package_files)

    native_libraries = sorted((package_root / "lib").glob("*.so*"))
    runtime_dir = package_root / "lib" / "torch_runtime"
    native_libraries.extend(sorted(runtime_dir.glob("*.so*")))
    native_libraries.extend(sorted(package_root.glob("_C*.so")))
    assert native_libraries, f"No native libraries found under {package_root}"
    for library in native_libraries:
        for search_path in _dynamic_search_paths(library):
            if require_isolation and vendor_root is not None:
                assert str(vendor_root) not in search_path, (
                    "Native library contains a vendor torch RPATH",
                    library,
                    search_path,
                )
            if require_isolation:
                assert "site-packages/torch/lib" not in search_path, (
                    "Native library contains an absolute Python torch RPATH",
                    library,
                    search_path,
                )

    if args.require_device:
        assert torch_fl.flagos.is_available(), "flagos device is unavailable"
        assert torch_fl.flagos.device_count() > 0, "No flagos devices were detected"

    package_metadata = metadata("torch_fl")
    print(f"Package: {package_metadata['Name']} {package_metadata['Version']}")
    print(f"torch: {torch.__version__} from {torch_root}")
    print(f"torch_fl: {package_root}")
    print(f"Bundled accelerator libraries: {len(libraries)}")
    if external_dependencies:
        print(
            "External accelerator runtime dependencies: "
            + ", ".join(sorted(external_dependencies))
        )
    if args.require_device:
        print(f"flagos devices: {torch_fl.flagos.device_count()}")


if __name__ == "__main__":
    main()
