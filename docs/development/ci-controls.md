# CI selection and maintainer controls

CI schedules nine configured hardware platforms through the shared integration
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

Routing labels take precedence over file filtering. Adding or removing labels
reruns PR CI. Supported platform names are `apple`, `ascend`, `enflame`, `hygon`,
`iluvatar`, `kunlunxin`, `metax`, `mthreads`, `nvidia` and `thead`.

Each `.github/configs/<platform>.yml` declares `paths_watch`: exact file names,
directory prefixes ending in `/`, or shell-style globs. Its own configuration
and dedicated workflow automatically belong to that lane. Shared code or an
unrecognized code path selects all lanes; a recognized vendor change must not
hide an unrelated shared change in the same PR. Missing or invalid watch
metadata also selects all lanes. A failed Git comparison selects all lanes.

Selection compares the immutable PR base and head commits and includes both
paths when files move between platforms. Watch lists include shared consumers:
CUDA runtime sources affect CUDA, DCU and PPU, while the CUDA allocator header
also affects MetaX. Keep these relationships aligned with
`cmake/flagos_platforms.json` and the runtime build configuration when adding or
changing a platform.

## Functional profiles

The default `pr` profile preserves the platform selection above. The daily
`nightly` profile runs every configured platform and all applicable existing
functional groups, including the dedicated CUDA transfer regressions. Manual
nightly requests require `platforms=all`. `weekly` is reserved and rejected
before hardware dispatch. Both CI and FlagCICD Entry expose these profile names.
See [Functional CI profiles](ci-test-tiers.md) for execution and evidence rules.

## One required check

The top-level job named **CI complete** runs after the shared checks and every
hardware pipeline. Shared checks and every selected platform must succeed.
Unselected platforms may skip. A selected platform that unexpectedly skips,
fails or is cancelled blocks the gate, as does missing or invalid discovery
output. A documentation-only PR can pass after its shared checks succeed.

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

### Implementation discipline

The existing integration PR tracks upstream main and its CI results. Planning
changes are developed on a dependent branch and reviewed in stages. When the
integration branch advances, update the dependent branch and recheck its own
delta; after the integration PR merges, align the planning branch with main.

For each stage, audit existing CI capabilities before adding code. Reuse the
router, composer, configuration and reporting facilities wherever they fit.
Existing facilities may be corrected or simplified when evidence shows a defect
or an unreasonable design; explain the impact and add relevant regression checks.
Keep one authoritative implementation for each responsibility. Observation beside
the existing execution path is transitional: when plan consumption and shared
gating take over, remove the superseded decision paths rather than retaining
parallel routing, composition or verdict implementations. New abstractions must
solve a demonstrated need. Each stage must include behavior checks and cleanup;
test exclusions or weaker gates must not hide defects.

### Observed execution plans

Ordinary CI retains `flagcicd-plan-<attempt>` after platform discovery. The
observation job runs beside hardware; neither hardware nor the required gate
depends on it, and its failures remain diagnostic warnings. Its regression tests
run when present in the tested checkout, so trusted dispatches to older source
commits do not fail because a newer test file is absent.

The existing matrix generator still supplies the hardware matrix, and `CI complete`
still uses the existing job/result gate. Plan generation and artifact upload are
non-blocking observation steps: a missing plan is unavailable diagnostic data,
never proof that tests ran or that a platform passed.

Both CI and FlagCICD Entry use `flagcicd_plan.py` to snapshot the final groups from
the existing manifest composer. There is no second group catalog or executor.
Each lane records its shared test environment and ordered groups, including the
original commands, group environments and `min_passed` requirements. The stable
`group_id` reuses the composer's `manifest_id`. Boxing plans use the same display
name and replacement environment expectations as `load-platform-config.yml`.
Shell variables remain unexpanded; runner environments and secret values are not
copied into the plan.

The plan records selection reasons and the actual expected shared checks.
Documentation PRs retain an empty hardware matrix and all three ordinary shared
checks. External requests keep their existing L0/L1/L2 policy, including the
explicit L2 empty-selection fallback. Neither policy is substituted for the other.

`plan_hash` binds the source repository/commit, comparison base and diff head,
control-code commit, profile, configuration identity and final execution content.
`source_sha` is the checked-out commit; `diff_head_sha` can instead be a PR head
when CI tests GitHub's merge commit. Run, attempt, task and parent-run identifiers
are excluded from this deterministic hash. The compact external report context
carries the hash, while commands and large routing explanations stay in the plan
artifact. The current report gate does not yet enforce this new hash.

This first stage does not make executors consume plans or add group routing,
report completeness gates, retries, caches, test skips or new hardware workloads.
Those changes require separate review and real-runner acceptance. Local plan
comparisons are not hardware validation.

### Functional evidence validation

The pytest merger, the ordinary hardware evidence check and the external platform
reporter share `validate_pytest_result`. Successful outer commands cannot hide a
nonzero pytest session exit code or failed/error counts. Conditional skipped
sessions remain valid when other functional tests passed; existing explicit
teardown-crash handling remains separate from assertion failures.

The ordinary gate reads paginated job diagnostics as a JSON array, with a bounded
API query. Optional diagnostic failures do not replace the required job verdicts.

Platform-agnostic CI runs regression tests for selection, routing, required gate
results, checkout resolution, comment dispatch, network classification and retry
budgets. GitHub operations in these tests are simulated; tests do not post real
comments or start hardware runs.

Before considering rollout verified, run hardware CI, configure the required
check, then exercise comment dispatch for an actual fork PR and observe a real
network-failure retry. The local tests and Actionlint checks cannot verify runner
availability, repository write permissions or GitHub's event delivery.
