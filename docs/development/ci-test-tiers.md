# CI test tiers

The tier contract lives in `.flagcicd/test-tiers.json`. Platform images, runner
labels, SDK setup and device expectations continue to live in
`.github/configs/<platform>.yml`. The integration manifest remains the ordered
test command inventory. Common workflows do not select chips in shell code.

## Profiles and scheduling

| Profile | Trigger | Selection |
| --- | --- | --- |
| `pr` | PR or push | Fixed L0/L1, then L2 on affected PR platforms or all push platforms |
| `pr-risk` | Runtime/build/compatibility changes | L2 plus bounded L3 compatibility, lifecycle, performance and applicable communication contracts |
| `nightly` | 02:00 CST Monday–Saturday | All configured platforms, L2 then L3 |
| `weekly` | 02:00 CST Sunday | Nightly selection plus 10-minute stability and available model acceptance workloads |
| `release` | Release tag workflow | Existing candidate wheel, L2, L3, mandatory model suites and at least one hour of stability |

FlagCICD Entry accepts `profile=auto/pr/pr-risk/nightly/weekly` and minimum
levels L0–L3. `auto` preserves L0 documentation checks and raises runtime
changes to the bounded L3 profile. Explicit levels cannot lower the diff's
minimum. Explicit nightly/weekly requests with `platforms=all` never narrow
the matrix using the diff. The plan records selected case IDs, levels,
categories, immutable source and a policy/configuration hash. Platform reports
must match the planned case sequence.

L0 validates configuration, lint and the explicit test file inventory. L1 runs
the existing agnostic, codegen and hardware-free wheel checks. Hardware jobs
depend on all of them. L2 cases precede L3 cases, and L3 precedes L4; the runner
stops on the first failure and retains already produced evidence. The existing
DCU source-contract sweep is diagnostic source testing; it does not establish
installed-wheel compatibility by itself and is excluded from candidate release
acceptance. Legacy unprofiled callers preserve
their command inventory; the main CI and unified entry use explicit profiles.

The L1 CUDA packaging build uses a separate Python 3.12 venv and the CPU PyTorch
pin from `.github/version-pins.env`; the container's preinstalled vendor Torch
is not a build dependency. The runner is pinned to Ubuntu 22.04 to match the
NGC 24.10 container: `setup-python` can reuse the host tool cache, whose Python
binaries must be compatible with the container's glibc. The job clears inherited
loader, linker and CMake search paths before `setup-python` adds its interpreter
library path; a venv alone does not isolate native libraries. The environment
probe verifies that loaded Torch core libraries come from the CPU installation.
Inherited pip configuration and extra indexes are disabled; build tools use
public PyPI and CPU Torch uses its explicit index. Boxing and FlagGems C++ are
disabled, matching the CUDA lane's Python-wrapper build. L1 checks metadata,
the installed files against the wheel archive, the native library and the
CPython extension filename from a clean target directory without importing
`torch_fl`. It does not claim
driver availability, full runtime
dependency installation, or accelerator execution; those checks belong to L2+.

Hardware job budgets are 60 minutes for PR profiles, six hours for nightly and
12 hours for weekly/release. Individual commands have explicit deadlines and
their process groups are killed on timeout. The weekly pipeline can consist of
multiple jobs; a two-day overall window is not a two-day single job. GitHub's
[token lifetime](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes)
also limits a job's ability to upload evidence after 24 hours.

## L3 and L4 suites

`tests/acceptance/` adds installed-package/dependency checks, fresh native runtime
startup, allocation/stream/backward/free/recovery, a fixed performance workload,
endurance and wrappers around the existing Qwen3 inference/training tests.
The integration manifest still provides native/FlagGems/fallback route contracts
and applicable multi-device/native communication contracts. Unsupported
multi-process or multi-card configurations do not become supported by adding a
profile. These suites require real hardware CI before any support claim changes.

The installed preflight compares the candidate build record to actual Python
distributions and ABI. Evidence includes the wheel compatibility record,
installed dependency versions, configured image, source SHA, hardware identity,
SDK and device count. Only the combinations represented by the current platform
configs are scheduled; this is not a claim that every Python/Torch/SDK cross
product is supported. Add another explicit platform/runtime configuration and
real runner evidence before widening the tested matrix.

Performance samples use `fp32-512x512-mul-add-sum-copy/v1`, five warmups and 20
timed repetitions. `.flagcicd/performance-baselines.json` keys approved thresholds
by build/device/Python identity. An entry requires `workload`, `sample_count`,
`reference_run`, `approved_by` and `max_median_seconds`. Nightly emits
`baseline_status=uncalibrated` when a reference is absent; release fails. Do not
invent a reference, automatically learn it from the candidate, or silently
increase a threshold to make a regression pass.

## Collection completeness

`.flagcicd/test-inventory.json` explicitly classifies every pytest-discoverable
file under `tests/`. Adding, moving or deleting a test requires updating its
owner, lane, fixed entrypoint and selector. L0 rejects unknown/stale files and
removed selectors. Existing files outside fixed commands are explicitly manual
gaps, rather than automatically included by a broad glob. Review those entries
before declaring full test coverage; registering a manual gap does not fix it.

At runtime, `groups.json` records the expected sequence, commands, levels,
categories and return codes, including groups that never started. The pytest
plugin records selected node IDs; merging requires a corresponding test result
for every selected item, a nonempty selection and a session for every pytest
group. Empty collection, missing reports, aborted sessions and unstarted groups
cannot produce complete evidence. Per-file source tests and nested model suites
retain their own sessions. Skipped/xfail cases retain their native reason.
Profiled runs disable historical teardown-crash tolerance.

## Candidate acceptance and publication

Release runs L0/L1 before building. It builds each candidate once and retains
`wheel-<platform>`. Acceptance downloads that artifact, provisions the runtime
without native rebuilding, validates and force-installs the wheel, then runs the
profile in a separate test workspace. The candidate receipt binds source, run,
platform, wheel filename/SHA-256, compatibility record, dependencies and
hardware. Installed package files are checked against the archive before and
after acceptance. Lower source checks are prerequisites at the same tag SHA;
candidate L2–L4 all use the exact same wheel.

Publication waits for every required platform's acceptance. A separate vendor
runner job downloads the original candidate and receipt, checks their identity
and hash, then calls the existing vendor uploader. Tested code receives no
publication credential. `build-wheel-common.yml` refuses direct `publish=true`
requests. A failed acceptance is not bypassed by manual workflow dispatch.

Release acceptance is deliberately strict: missing hardware/SDK identity,
unconfigured model acceptance, missing performance references, skipped/xfail
cases, crashes or incomplete evidence block publication. There is currently no
approved skip allowlist. The initial model workload is configured only for CUDA
and PPU; the other publishing platforms need owned model mounts and verified
workloads. No performance references have been fabricated. Therefore this
branch does **not** yet establish any platform's L4 acceptance or allow a full
seven-platform release to pass without that follow-up evidence.

Inspect `acceptance-<platform>`, `pytest-<platform>-<attempt>` and
`flagcicd-plan-<attempt>` artifacts. Weekly results are regression evidence, not
release approval. After changing source, dependencies, images, model data or the
candidate bytes, build a new candidate and rerun the affected acceptance. For a
failed release, rerun failed jobs to preserve successful candidate artifacts and
receipts. Receipts may come from an earlier attempt of the same run when the
source SHA and wheel hash still match; they never come from a different run. An immutable package index may still have partial uploads if the
network fails after acceptance; follow the release recovery procedure without
rebuilding or changing the accepted wheel.
