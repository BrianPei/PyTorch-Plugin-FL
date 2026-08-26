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

Two classes of dependency, because they fail differently:

--require  the test harness itself (pytest) and the model stack the inference and
           training groups load (transformers, safetensors). Missing means no
           group can produce meaningful evidence, so this exits non-zero and the
           job stops in setup with an unambiguous message.

--expect   a capability a single group needs, typically the vendor Triton behind
           torch.compile. Missing is reported as a warning and the run continues:
           the group that needs it fails on its own and that failure is the
           record for the platform owners, while the other groups still run.
           Exiting here would erase all of that evidence over one gap.

For triton specifically, --expect-triton-verbose requests a deeper probe:
importability, version, backend registry, and the path it resolves to. Stock
PyPI triton can be imported but has 0 active drivers, so an import-only check
would pass on the wrong Triton. The verbose check exposes that gap while still
treating the absence as a warning rather than a hard failure.

Usage:
    python .github/scripts/check_integration_deps.py --require pytest transformers
    python .github/scripts/check_integration_deps.py --expect triton \
        --expect-hint "vendor triton-ascend provides torch.compile on this line"
    python .github/scripts/check_integration_deps.py --expect triton \
        --expect-triton-verbose \
        --expect-hint "vendor triton-ascend provides torch.compile on this line"
"""

import argparse
import importlib.util
import sys


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


def _report_present(label, names, missing):
    present = [n for n in names if n not in missing]
    if present:
        print(f"{label} present: {', '.join(present)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require",
        nargs="*",
        default=[],
        metavar="MODULE",
        help="Modules whose absence is a setup failure (exit non-zero)",
    )
    parser.add_argument(
        "--expect",
        nargs="*",
        default=[],
        metavar="MODULE",
        help="Modules whose absence is a warning; the dependent group fails on its own",
    )
    parser.add_argument(
        "--expect-hint",
        default="",
        help="Text appended to the warning, naming the vendor package to install",
    )
    parser.add_argument(
        "--expect-triton-verbose",
        action="store_true",
        help="When triton is in --expect, probe version/backends/path in addition to importability",
    )
    args = parser.parse_args()

    print(f"Dependency check interpreter: {sys.executable}")

    expected_missing = _missing(args.expect)
    _report_present("Expected", args.expect, expected_missing)

    # Verbose triton check: import it, show version/backends/path, warn if 0 drivers
    if (
        "triton" in args.expect
        and "triton" not in expected_missing
        and args.expect_triton_verbose
    ):
        try:
            import triton

            print(f"triton: {triton.__file__}")
            print(f"triton version: {triton.__version__}")
            # triton.runtime.driver exposes get_active_drivers() or get_drivers()
            # depending on version; stock PyPI triton returns an empty list
            if hasattr(triton.runtime, "driver"):
                drivers = getattr(
                    triton.runtime.driver, "get_active_drivers", lambda: []
                )()
                if not drivers:
                    drivers = getattr(
                        triton.runtime.driver, "get_drivers", lambda: []
                    )()
                print(f"triton active drivers: {drivers if drivers else '(none)'}")
                if not drivers:
                    print(
                        "::warning::triton is importable but has 0 active drivers; "
                        "stock PyPI triton is not a substitute for a vendor build "
                        "(triton-ascend, triton-metax, flagtree). torch.compile will fail."
                    )
            else:
                print("triton.runtime.driver not found; cannot probe backends")
        except Exception as error:
            print(f"::warning::triton import succeeded but inspection failed: {error}")

    if expected_missing:
        hint = f" {args.expect_hint}" if args.expect_hint else ""
        print(
            f"::warning::Not importable: {', '.join(expected_missing)}."
            f"{hint} The group that needs it will fail, and that failure is the"
            " record; the remaining groups still run."
        )

    required_missing = _missing(args.require)
    _report_present("Required", args.require, required_missing)
    if required_missing:
        print(
            f"::error::The test interpreter cannot import: {', '.join(required_missing)}."
            " Install these in the platform setup script or bake them into the CI"
            " image; without them the functional groups report an environment"
            " problem as a platform problem."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
