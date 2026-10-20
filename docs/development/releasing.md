# Releasing

A release starts from a version tag and must pass candidate-wheel acceptance
before any vendor lane is published. Both stable and prerelease tags use the
same gate. Manual dispatch must select the tag ref and cannot bypass acceptance.
See [CI test tiers](ci-test-tiers.md) for profiles, evidence and remaining gaps.

## Release sequence

1. `release.yml` checks the tag, lint/configuration, hardware-free tests, codegen
   and L1 wheel checks at the tag SHA.
2. `build-wheel.yml` builds one candidate on each publishing platform's runner
   using its existing platform config. `publish` is false. The resulting
   `wheel-<platform>` artifacts are the immutable candidates for this run.
3. The release profile downloads each candidate, provisions the runtime without
   rebuilding native code, validates and installs it, and runs candidate L2–L4.
   Acceptance compares installed package files with the wheel archive and
   retains its SHA-256, source, build record, dependencies, hardware/SDK and
   collected/executed test inventory in `acceptance-<platform>`.
4. Only after every required platform accepts its candidate does
   `publish-accepted-wheel.yml` run. It downloads the same candidate bytes and
   receipt on each vendor runner, verifies the source/run/platform/hash and
   accepted status, then calls `publish_wheels.py`. No wheel is rebuilt here.

Required platforms are ascend, cuda, dcu, gcu, metax, musa and ppu. Their vendor
lanes remain defined by `cmake/flagos_platforms.json`. Runner labels, images and
mounts remain in `.github/configs/`. Publication still occurs on vendor runners,
so package-index upload does not move to a GitHub-hosted runner. Artifact
transfer now necessarily precedes acceptance and publication; a transfer failure
blocks the release rather than bypassing evidence.

## Acceptance prerequisites

The branch does not yet contain approved performance reference measurements or
verified model workloads for every publishing platform. CUDA and PPU have
existing Qwen3 model commands; others must supply owned model mounts and verified
workloads. Release acceptance also requires known SDK provenance, actual device
identity and no unapproved skipped/xfail cases. Until those prerequisites pass on
real hardware, a complete seven-platform release is intentionally blocked.
Weekly regression results alone do not approve a candidate.

Adding a publishing platform requires its platform table/config, candidate
build, acceptance and publish matrix entries, model/workload configuration,
performance reference and actual hardware evidence. All three release matrices
must match the publishing lanes. A passing source test is insufficient.

## Credentials and uploader behavior

`NEXUS_TOKEN` is the organization secret in `user:token` form. It is explicitly
mapped only through the publishing workflow in the release graph; candidate
testing receives no publishing credential. Direct `publish=true` requests to
`build-wheel-common.yml` fail and explain the required release path.

The existing uploader still verifies that the wheel base version matches the
tag, reads vendor lanes from the platform table, prefers an available direct or
proxy route, and retries bounded transient transport failures. Already published
versions are skipped. The package index does not overwrite an existing version.

## Recovering a failed release

Inspect the failed stage and its `acceptance-<platform>`, pytest and wheel
artifacts. Fix an environment/mount/transport problem without changing the
candidate, then rerun failed jobs. Successful build artifacts remain unchanged.
Successful receipts from an earlier attempt of the same run remain usable only
when source SHA, platform, wheel filename and SHA-256 still match. Failed
acceptance cannot be bypassed by dispatch or a direct build-workflow publish.

After all acceptance passes, a publication network failure can still leave some
lanes already uploaded. Rerun failed publish jobs with the original candidates
and accepted receipts; do not rebuild that version. If source, package bytes,
dependencies or acceptance policy must change, cut a new candidate/version and
run acceptance again. There is no atomic transaction across vendor lanes, and a
partially uploaded immutable version cannot be repaired by replacing its bytes.
