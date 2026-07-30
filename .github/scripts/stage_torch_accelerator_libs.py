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

"""Stage accelerator-side PyTorch libraries for inclusion in the torch_fl wheel.

The build environment supplies two different PyTorch installations:

* upstream CPU PyTorch provides Python, headers, and the common ATen libraries;
* a vendor PyTorch image provides only the accelerator-side shared libraries.

This script copies the vendor libraries and their source-local DT_NEEDED closure
into ``torch_fl/lib/torch_runtime``.  Common PyTorch libraries are deliberately
excluded so that they continue to come from the upstream CPU wheel at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


DEFAULT_SEEDS = (
    "libc10_cuda.so",
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
)

# These libraries must be supplied by the installed upstream CPU PyTorch wheel.
COMMON_TORCH_LIBRARIES = {
    "libc10.so",
    "libshm.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
}

_DYNAMIC_ENTRY_RE = re.compile(
    r"\((?P<kind>NEEDED|SONAME)\).*?: \[(?P<name>[^\]]+)\]"
)


def _dynamic_entries(path: Path) -> tuple[list[str], str | None]:
    try:
        output = subprocess.check_output(
            ["readelf", "-d", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as error:
        raise RuntimeError("readelf is required to stage vendor libraries") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"readelf failed for {path}:\n{error.output}") from error

    needed: list[str] = []
    soname: str | None = None
    for line in output.splitlines():
        match = _DYNAMIC_ENTRY_RE.search(line)
        if not match:
            continue
        if match.group("kind") == "NEEDED":
            needed.append(match.group("name"))
        else:
            soname = match.group("name")
    return needed, soname


def _library_index(source: Path) -> tuple[dict[str, Path], dict[Path, list[str]]]:
    by_name: dict[str, Path] = {}
    names_by_file: dict[Path, list[str]] = defaultdict(list)

    for candidate in sorted(source.glob("*.so*")):
        if not candidate.is_file():
            continue
        real_file = candidate.resolve()
        by_name.setdefault(candidate.name, candidate)
        names_by_file[real_file].append(candidate.name)

    for real_file, names in names_by_file.items():
        _, soname = _dynamic_entries(real_file)
        if soname:
            by_name.setdefault(soname, real_file)
            names.append(soname)

    return by_name, names_by_file


def _resolve_seed(seed: str, source: Path, by_name: dict[str, Path]) -> Path | None:
    exact = by_name.get(seed)
    if exact is not None:
        return exact
    matches = sorted(
        (path for path in source.glob(f"{seed}*") if path.is_file()),
        key=lambda path: (len(path.name), path.name),
    )
    return matches[0] if matches else None


def _is_common_torch_library(name: str) -> bool:
    return name in COMMON_TORCH_LIBRARIES or any(
        name.startswith(f"{prefix}.") for prefix in COMMON_TORCH_LIBRARIES
    )


def _load_order(graph: dict[str, list[str]]) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        # ELF dependency cycles are legal. The dynamic linker resolves them
        # when the first member is loaded, so stop descending on a back edge.
        if name in visiting:
            return
        visiting.add(name)
        for dependency in graph.get(name, []):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for library in graph:
        visit(library)
    return ordered


def stage_libraries(source: Path, destination: Path, seeds: list[str]) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise RuntimeError(f"vendor torch library directory does not exist: {source}")
    if source == destination:
        raise RuntimeError("source and destination must be different directories")

    by_name, _ = _library_index(source)
    resolved_seeds = {
        seed: path
        for seed in seeds
        if (path := _resolve_seed(seed, source, by_name)) is not None
    }
    if "libtorch_cuda.so" not in resolved_seeds:
        raise RuntimeError(
            f"libtorch_cuda.so was not found under vendor torch directory {source}"
        )

    pending: list[tuple[str, Path]] = list(resolved_seeds.items())
    selected: dict[str, Path] = {}
    dependency_graph: dict[str, list[str]] = {}
    external_dependencies: set[str] = set()
    common_dependencies: set[str] = set()

    while pending:
        install_name, source_path = pending.pop(0)
        if install_name in selected:
            continue
        source_path = source_path.resolve()
        selected[install_name] = source_path

        needed, _ = _dynamic_entries(source_path)
        bundled_dependencies: list[str] = []
        for dependency in needed:
            if _is_common_torch_library(dependency):
                common_dependencies.add(dependency)
                continue
            dependency_path = by_name.get(dependency)
            if dependency_path is None:
                external_dependencies.add(dependency)
                continue
            bundled_dependencies.append(dependency)
            pending.append((dependency, dependency_path))
        dependency_graph[install_name] = bundled_dependencies

    destination.mkdir(parents=True, exist_ok=True)
    for child in destination.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for install_name, source_path in sorted(selected.items()):
        shutil.copy2(source_path, destination / install_name)

    manifest = {
        "schema_version": 1,
        "libraries": sorted(selected),
        "preload": [],
        "load_order": _load_order(dependency_graph),
        "common_torch_dependencies": sorted(common_dependencies),
        "external_dependencies": sorted(external_dependencies),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="vendor Python torch/lib directory",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("torch_fl/lib/torch_runtime"),
        help="wheel staging directory",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        help="accelerator library name; repeat to override the default seed set",
    )
    args = parser.parse_args()

    manifest = stage_libraries(
        args.source,
        args.destination,
        args.seeds or list(DEFAULT_SEEDS),
    )
    print(
        f"Staged {len(manifest['libraries'])} vendor torch libraries in "
        f"{args.destination}"
    )
    for library in manifest["load_order"]:
        print(f"  {library}")
    if manifest["external_dependencies"]:
        print("External runtime dependencies:")
        for dependency in manifest["external_dependencies"]:
            print(f"  {dependency}")


if __name__ == "__main__":
    main()
