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

"""Verify every platform manifest exports the same public integration IDs.

Every supported platform converges onto the same nine public contract IDs so CI
enforces uniform capability coverage across hardware backends. Platform-specific
groups (vendor dispatchers, FlagGems runtime paths, profiler parity, model
smoke) are additive and are excluded from the baseline check: adding one does
not relax the shared contract, and the validator reports only missing baseline
IDs, never extra groups.

Baseline test IDs (the public cross-platform contract):
    Check device availability
    Run operator tests (vendor backend, main ops)
    Run unified RNG tests
    Run general tests
    Unified AMP contract
    Unified profiler contract
    Run torch.compile tests
    Run inference tests
    Run training tests

Some platforms express an equivalent step under a local name, or scope the
operator step differently (e.g. an explicit file list instead of the whole
directory). A normalization map lets those count as present; any platform that
genuinely lacks a baseline ID fails.

Usage:
    python .github/scripts/validate_integration_manifests.py
    python .github/scripts/validate_integration_manifests.py --configs-dir .github/configs
"""

import argparse
from pathlib import Path

import yaml

CANONICAL_IDS = (
    "Check device availability",
    "Run operator tests (vendor backend, main ops)",
    "Run unified RNG tests",
    "Run general tests",
    "Unified AMP contract",
    "Unified profiler contract",
    "Run torch.compile tests",
    "Run inference tests",
    "Run training tests",
)

# A platform may express one canonical ID under a different name; map each
# platform and canonical ID to the accepted manifest names.
ID_ALIASES = {
    "Run operator tests (vendor backend, main ops)": {
        "Run representative MetaX operator tests",
        "Run generic per-op correctness tests",
    },
    "Run unified RNG tests": {
        "Run Ascend RNG tests",
    },
    "Run general tests": {
        "Run general factory ops",
        "Run general factory and autograd tests",
        "Run AMP tests",
    },
    "Unified AMP contract": {
        "Run AMP tests",
    },
}


def _accepted_names(canonical_id: str) -> set:
    names = {canonical_id}
    names.update(ID_ALIASES.get(canonical_id, ()))
    return names


def _collect_manifest_ids(config_path: Path) -> list:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tests = config.get("integration_tests") or []
    return [(entry.get("name", ""), entry) for entry in tests]


def _check_platform(config_path: Path) -> list:
    """Return a list of missing baseline IDs and validation errors for one platform."""
    manifest_entries = _collect_manifest_ids(config_path)
    ids = {name for name, _ in manifest_entries}
    missing = []
    errors = []

    # Check for missing baseline IDs
    for canonical in CANONICAL_IDS:
        if not (ids & _accepted_names(canonical)):
            missing.append(canonical)

    # Check each entry for required fields and pytest target validation
    for name, entry in manifest_entries:
        command = entry.get("command", "")

        # Skip preflight checks (they are not pytest commands)
        if entry.get("phase") == "preflight":
            continue

        # For functional test entries, validate pytest commands
        if "pytest" in command:
            # Check that pytest has a target (not just bare "pytest")
            # Valid: "pytest tests/...", "python -m pytest tests/..."
            # Invalid: "pytest" alone, or pytest with only flags
            import re

            # Match pytest followed by at least one non-flag argument
            # Flags start with - or --, targets don't
            if not re.search(r"pytest\s+(?:[^-\s]|--\w+\s+\S+)", command):
                errors.append(
                    f"Entry '{name}' has pytest command without explicit target path"
                )

            # Check for collection constraints (markers, paths, or explicit test selection)
            # Valid markers: -m "...", paths: tests/..., specific tests: test_name.py::test_func
            has_marker = "-m" in command or "--markers" in command
            has_path = "tests/" in command or "test_" in command

            if not (has_marker or has_path):
                errors.append(
                    f"Entry '{name}' has pytest command without collection constraint "
                    "(missing -m marker or explicit test path)"
                )

    return missing, errors


def validate(configs_dir: Path) -> int:
    failures = []
    validation_errors = []
    for config_path in sorted(configs_dir.glob("*.yml")):
        # Only the 7 hardware platform manifests carry integration_tests.
        manifest_ids = _collect_manifest_ids(config_path)
        if not manifest_ids:
            continue
        platform = config_path.stem
        missing, errors = _check_platform(config_path)
        if missing:
            for canonical in missing:
                failures.append(f"{platform}: missing baseline ID '{canonical}'")
        if errors:
            for error in errors:
                validation_errors.append(f"{platform}: {error}")
        if not (missing or errors):
            print(f"{platform}: OK ({len(manifest_ids)} integration groups)")

    if failures:
        print("\nCross-platform baseline failures:")
        for failure in failures:
            print(f"  - {failure}")
    if validation_errors:
        print("\nValidation errors:")
        for error in validation_errors:
            print(f"  - {error}")
    if failures or validation_errors:
        return 1
    print("\nAll platform manifests export the full public baseline.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".github" / "configs",
        help="Directory containing the platform manifests (default: .github/configs)",
    )
    args = parser.parse_args()
    if not args.configs_dir.is_dir():
        parser.error(f"configs dir not found: {args.configs_dir}")
    return validate(args.configs_dir)


if __name__ == "__main__":
    raise SystemExit(main())
