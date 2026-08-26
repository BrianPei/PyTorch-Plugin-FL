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

"""Verify the interpreter that will run the tests can import what they need.

Called from every platform's set_env script, including when it adopted a
prebuilt venv. A venv baked without pytest or transformers turns the functional
groups into an environment failure wearing a platform failure's clothes, and the
resulting log blames the vendor backend for a missing pip install.

All dependencies declared with --require are mandatory: missing or incorrect
means the job stops in setup with an unambiguous message. There is no
warning-only mode.

For triton specifically, --triton-backend requests a full validation:
importability, version, backend registry key, and driver availability. Stock
PyPI triton can be imported but has 0 active drivers, so an import-only check
would pass on the wrong Triton. This check fails if triton is absent, has no
drivers, or is not the vendor build.

Usage:
    python .github/scripts/check_integration_deps.py --require pytest transformers safetensors
    python .github/scripts/check_integration_deps.py --require triton --triton-backend ascend
"""

import argparse
import importlib.util
import sys


def _check_triton_backend(backend_name):
    """Validate vendor triton: version, backend registry, and driver availability.

    Returns (success, errors) where errors is a list of validation failures.
    Stock PyPI triton has 0 drivers and is not a substitute for vendor builds.
    """
    errors = []
    try:
        import triton
    except ImportError as e:
        return False, [f"triton import failed: {e}"]

    print(f"triton.__file__ = {triton.__file__}")
    print(f"triton.__version__ = {triton.__version__}")

    # Check backend registry
    if not hasattr(triton.runtime, "driver"):
        errors.append("triton.runtime.driver not found; cannot validate backend")
        return False, errors

    drivers = getattr(triton.runtime.driver, "get_active_drivers", lambda: [])()
    if not drivers:
        drivers = getattr(triton.runtime.driver, "get_drivers", lambda: [])()

    print(f"triton active drivers: {drivers if drivers else '(none)'}")

    if not drivers:
        errors.append(
            "triton has 0 active drivers; stock PyPI triton is not a substitute "
            f"for vendor build (triton-{backend_name} required)"
        )
        return False, errors

    # Validate the expected backend is registered
    expected_key = backend_name.lower()
    if expected_key not in [d.lower() for d in drivers]:
        errors.append(
            f"expected backend '{expected_key}' not in active drivers: {drivers}"
        )
        return False, errors

    print(f"triton backend '{expected_key}' validated successfully")
    return True, []


def _missing(names):
    missing = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A package whose parent is absent, or a name that cannot be a spec.
            found = False
        if not found:
            missing.append(name)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require",
        nargs="*",
        default=[],
        metavar="MODULE",
        help="Modules that must be importable (exit non-zero if absent)",
    )
    parser.add_argument(
        "--triton-backend",
        metavar="NAME",
        help="Validate triton with the specified backend (ascend, metax, cuda, etc.)",
    )
    args = parser.parse_args()

    print(f"Dependency check interpreter: {sys.executable}")

    all_errors = []

    # Check required modules
    required_missing = _missing(args.require)
    present = [n for n in args.require if n not in required_missing]
    if present:
        print(f"Required modules present: {', '.join(present)}")

    if required_missing:
        all_errors.append(
            f"Cannot import required modules: {', '.join(required_missing)}"
        )

    # Full triton validation if requested
    if args.triton_backend:
        if "triton" in required_missing:
            all_errors.append(
                f"triton is required but not importable; install triton-{args.triton_backend}"
            )
        else:
            success, triton_errors = _check_triton_backend(args.triton_backend)
            if not success:
                all_errors.extend(triton_errors)

    if all_errors:
        print("\n=== Dependency Check Failed ===")
        for error in all_errors:
            print(f"::error::{error}")
        print(
            "\nInstall missing dependencies in the platform setup script or bake them "
            "into the CI image."
        )
        return 1

    print("\n✓ All dependency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
