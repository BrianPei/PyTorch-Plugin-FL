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
| `nightly` | 02:00 CST daily | All configured platforms and every applicable L0-L4 case, including endurance and configured model workloads |
| `weekly` | Manual/future profile | Reserved for a separately designed weekly stability and model schedule; it is not automatically triggered |
| `release` | Release tag workflow | Existing candidate wheel, L2, L3, mandatory model suites and at least one hour of stability |

FlagCICD Entry accepts `profile=auto/pr/pr-risk/nightly/weekly` and minimum
levels L0–L3. `auto` preserves L0 documentation checks and raises runtime
changes to the bounded L3 profile. Explicit levels cannot lower the diff's
minimum. Explicit nightly/weekly requests with `platforms=all` never narrow
the matrix using the diff. The plan records selected case IDs, levels,
categories, immutable source and a policy/configuration hash. Platform reports
must match the planned case sequence.

The automatic CI and external entry share request resolution, planning and
aggregation. Event adapters preserve existing selection defaults and record
`selection=affected/all` separately from the platform list. Automatic PRs keep
their L1 shared-check minimum, and manual CI requests keep a hardware minimum.
The plan distinguishes the tested source (`source_sha`), comparison head
(`diff_sha`), head/merge target kind and immutable control-code version.
Profile and control identity are included in both the fingerprint and report
matching, including when successful platform results survive a failed-job rerun.

L0 validates configuration, lint and the explicit test file inventory. L1 runs
the existing agnostic, codegen and hardware-free wheel checks. Hardware jobs
depend on all of them. L2 cases precede L3 cases, and L3 precedes L4.

Setup, build, preflight and wheel installation run once per hardware job; only
the test groups after them run concurrently. Consecutive single-accelerator
L2/L3 groups that explicitly declare `execution: parallel` and
`resources.shared: true` form a shared batch with at most
`FLAGCICD_MAX_PARALLEL` (default 2, set by the workflow; 1 restores a serial
run) groups at a time, each in its own process with whole, `[group]`-prefixed
output lines. An `exclusive` group, a group needing more than one accelerator
and every L4 group runs alone, so it never overlaps another group and the
manifest order across it is kept. Environment and runtime preflights,
profiler suites (device tracers are process-wide), multi-device and
communication suites, compile, inference, training, installed-wheel,
lifecycle, performance and model cases are exclusive. After a failure no new
group starts: in-flight commands in that shared batch are terminated and keep
their terminal evidence, every unstarted group is recorded as `not_run`, and
the job exits with the first observed root failure rather than a sibling's
fail-fast `SIGKILL`. Cancellation kills every running group. The existing
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
`torch_fl`; build metadata in the checkout is removed before the installed
distribution is resolved. It does not claim
driver availability, full runtime
dependency installation, or accelerator execution; those checks belong to L2+.

Hardware job budgets are 60 minutes for PR profiles and 12 hours for nightly,
weekly and release. Individual commands have explicit deadlines and
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

Each stable case ID has a level, explicit `profiles`, `requires`,
`resources.min_accelerators`, `timeout_seconds` and an explicit `execution`
mode in `test-tiers.json`. L2/L3 cases must declare either `parallel` with
`resources.shared: true` or `exclusive`; L4 cases are exclusive. A case that
holds device-wide or host-wide state (a tracer, several devices, a model, a
cache it measures) declares `exclusive` explicitly.
Commands remain solely in the integration manifest (acceptance commands are
assembled by the existing tier selector). Requirements currently cover an
accelerator, a model directory, a metrics directory and the primary device.
The composer reads the existing
`environment_check` contract; it does not create a second platform support map.
An undeclared required model directory or an insufficient exact device count
is recorded as blocked, with the group still required by the plan. Other
device counts require runtime verification, rather than being claimed as
measured hardware. Multi-device groups require two devices; existing L2 groups
and the DCU native bridge's world-size-one packaging/runtime contract require
one. Performance/endurance/model acceptance groups require
`FLAGCICD_METRICS_DIR` and recorded `current_device == 0`; the runner records
that actual current device in the hardware identity and blocks when the
requirement is absent or mismatched. Resource budgets follow the actual
workload, rather than its category.

The runner checks `hardware-<platform>.json` before executing each profiled
group. Missing, invalid or insufficient hardware evidence blocks the group and
retains its reason in `groups.json`; it never turns into a passing skip. A
missing metrics directory or a non-primary current device is handled the same
way. Acceptance extras stream through the same runner with a group prefix so
long model/performance commands remain visible. The platform report preserves
these details as an infrastructure failure. Summary
artifacts include platform decisions and a reference to the complete plan
artifact, where all per-case exclusion and capability decisions are retained.

`.flagcicd/test-inventory.json` explicitly classifies every pytest-discoverable
file under `tests/`. Adding, moving or deleting a test requires updating its
owner, lane, fixed entrypoint and selector. L0 rejects unknown/stale files and
removed selectors. Existing files outside fixed commands are explicitly manual
gaps, rather than automatically included by a broad glob. Review those entries
before declaring full test coverage; registering a manual gap does not fix it.
Manual entries also require a disposition. The 31 existing gaps remain
`awaiting_platform_validation`, with their existing owner and individual
reason; the plan lists them separately and the summary records their count.

At runtime, `groups.json` records the expected sequence, commands, levels,
categories, execution modes and return codes, including groups that never
started. The pytest
plugin records selected node IDs; merging requires a corresponding test result
for every selected item, a nonempty selection and a session for every pytest
group. Empty collection, missing reports, aborted sessions and unstarted groups
cannot produce complete evidence. Per-file source tests and nested model suites
retain their own sessions. Skipped/xfail cases retain their native reason.
Profiled runs disable historical teardown-crash tolerance.
Profiled pytest groups each require at least one passing test. A different
passing group cannot hide a required group whose tests all skipped or xfailed.
Tier policy may explicitly declare a platform unsupported for a case, with a
nonempty reason. Such cases remain visible as `unsupported` in the plan and
in `unsupported_cases` in platform and aggregate reports; they are never counted
as passing tests. MPS exposes one logical device, so its cross-device case is
declared unsupported. Other platforms retain the two-device requirement and
remain blocked when their recorded hardware is insufficient. This declaration
does not establish multi-device support on MPS or relax release model gates.
The ledger includes selection reasons, resource requirements, start/end times,
duration, exit code and run log URL; merged execution evidence adds group test
counts and pytest session paths. An incomplete failed run still retains its
observed counts without making an acceptance claim.

The compact plan context also carries each stable case's minimum passing-test
count and accelerator count. The platform reporter independently compares
these requirements against every executed group and the recorded hardware;
an asserted `complete=true` or a passing grand total cannot override a
contradictory group. Malformed execution evidence produces a retained
`report_error` rather than preventing the platform report from being written.

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
