# CI selection and maintainer controls

CI schedules configured hardware platforms through the shared integration
workflow. Runner labels, images and device access come from each platform config;
test groups come from the shared integration manifest.

## Automatic platform selection

| Event | Hardware selection |
| --- | --- |
| PR with `platform:all` | Every configured platform |
| PR with one or more `platform:<name>` labels | The union of those platforms |
| PR with an unknown routing label | Every platform, conservatively |
| PR without routing labels | Platforms affected by the changed files |
| Documentation-only PR without routing labels | No hardware jobs; shared checks still run |
| Push or daily schedule | Every platform |
| Manual dispatch | `platforms=all` or a comma-separated platform list |

Automatic CI and FlagCICD Entry normalize requests through `flagcicd_resolve.py`,
then use the same `flagcicd_plan.py` planner and `flagcicd_report.py` aggregator.
Both retain `flagcicd-plan-<attempt>`, per-platform outcomes and a
`flagcicd-run-summary-<attempt>` artifact. Automatic documentation PRs preserve
the existing shared L1 checks without hardware; an external automatic-level
documentation request can remain L0. CI manual dispatch preserves a hardware
minimum and interprets `platforms=all` as the full matrix. Legacy external
`platforms=all` remains change-aware unless an explicit full selection or
nightly/weekly profile is requested.

Ordinary PR CI tests its event merge commit and records `target_kind=merge`;
selection compares the PR base with its head via `diff_sha`. An on-demand fork
PR request resolves and tests the immutable head with `target_kind=head`.
`control_sha` records the workflow's control-code version independently of the
source under test. The plan fingerprint includes execution profile, expected
cases and control identity so a rerun cannot reuse evidence for another policy.

Routing labels take precedence over file filtering. Adding or removing labels
reruns PR CI. Supported platform names are `ascend`, `cuda`, `dcu`, `gcu`,
`iluvatar`, `kunlunxin`, `metax`, `mps`, `musa` and `ppu`.

Each `.github/configs/<platform>.yml` declares `paths_watch`: exact file names,
directory prefixes ending in `/`, or shell-style globs. Its own configuration
and dedicated workflow automatically belong to that lane. Shared code or an
unrecognized code path selects all lanes; a recognized vendor change must not
hide an unrelated shared change in the same PR. Missing or invalid watch
metadata also selects all lanes. A failed Git comparison selects all lanes.
Fetching a comparison commit has a bounded, non-fatal step: if it remains
unavailable, the shared planner records `diff_status=unavailable` and uses its
conservative selection policy instead of stopping before that fallback.

Selection compares the immutable PR base and head commits and includes both
paths when files move between platforms. Watch lists include shared consumers:
CUDA runtime sources affect CUDA, DCU and PPU, while the CUDA allocator header
also affects MetaX. Keep these relationships aligned with
`cmake/flagos_platforms.json` and the runtime build configuration when adding or
changing a platform.

## One required check

The top-level job named **CI complete** runs after the shared checks and every
hardware pipeline. Shared checks and every selected platform must succeed.
Unselected platforms may skip. A selected platform that unexpectedly skips,
fails or is cancelled blocks the gate, as do an unavailable plan, missing or
contradictory platform reports, incomplete test collection and a run with no
passing pytest cases. The gate checks planned case IDs against execution
evidence instead of trusting the matrix job result alone. A documentation-only
PR can pass after its shared checks and configuration validation succeed.

A repository administrator can make **CI complete** the required status check
in the branch protection rule or ruleset. Adding the job does not change that
repository setting automatically. Do not substitute individual platform job
names for this aggregate check.

## `/ci` comments

Repository writers, maintainers and administrators can post a command on the
first line of a PR comment:

```text
/ci run
/ci run all
/ci run ascend
/ci run ascend cuda
/ci run ascend,cuda
```

The command validates platform names, keeps unrelated labels, replaces routing
labels and dispatches `ci.yml` on the base repository's default branch. That
trusted workflow resolves the open PR's current head repository and immutable
commit; every shared check, hardware checkout and manifest loader receives the
same source. This supports fork PRs without loading workflow definitions from
the fork. Authoring a PR alone does not authorize this privileged hardware
dispatch route.

A valid command is acknowledged only after dispatch succeeds. Invalid commands
from authorized users receive a usage response. Closed PRs and deleted fork
heads cannot dispatch. Separate concurrency groups keep a requested PR run from
cancelling main, nightly CI or another PR's requested run.

Requested runs provide diagnostics for the PR head. Their check run belongs to
the dispatch ref; they do not replace the ordinary PR's required **CI complete**
check or publish a synthetic passing check on the PR commit.

## Bounded network retry

`flaky-retry.yml` listens for completed, failed runs of `ci.yml`. It checks the
actual run and all failed root jobs using the GitHub API. It excludes the
dependent **CI complete** summary from root-cause classification.

Only recognized network failures are retried: for example pip read timeouts,
DNS failures, connection resets, selected curl transport failures and HTTP
502/503/504 responses. Every failed root job must qualify. Assertion, import,
compiler, missing-package and OOM failures take precedence over network noise.
Unknown failures, driver faults, collective hangs, plain timeouts, unavailable
logs and cancellations remain failed for investigation.

The platform outcome job records failed execution without failing merely because
that recorded verdict is non-passing. The hardware job retains its original
failure and **CI complete** enforces the report verdict. This keeps report-only
bookkeeping from becoming an extra unknown root failure that disables the
bounded network retry. A report generation or artifact transfer failure still
fails normally, and the summary command cannot use record-only mode.
A non-passing report after otherwise successful hardware also fails the outcome
job, so rerunning failed jobs regenerates that report before recalculating the
summary. Invalid stage data is retained as an explicit error in the report.

The policy requests `gh run rerun --failed` at most once, including dependent
jobs so the aggregate gate is recalculated. It checks the current attempt again
immediately before requesting a rerun; a duplicate event or manual rerun must
not consume another retry. GitHub CLI calls have a 120-second limit and the
supplemental workflow has a five-minute limit. Retry does not turn a failed CI
result into a pass: the rerun must succeed. No AI service credentials are needed.

Privileged control workflows checkout scripts from their own trusted event
commit and never execute a failed PR's scripts or downloaded artifacts. Both
[`issue_comment`](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment)
and
[`workflow_run`](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
require their workflow definitions to exist on the default branch.

## Validation and rollout

Platform-agnostic CI runs regression tests for selection, routing, required gate
results, checkout resolution, comment dispatch, network classification and retry
budgets. GitHub operations in these tests are simulated; tests do not post real
comments or start hardware runs.

Before considering rollout verified, run hardware CI, configure the required
check, then exercise comment dispatch for an actual fork PR and observe a real
network-failure retry. The local tests and Actionlint checks cannot verify runner
availability, repository write permissions or GitHub's event delivery.
