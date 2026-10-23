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

Required CI lanes are ascend, nvidia, hygon, enflame, metax, mthreads and thead. Their vendor
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

The current publishing lanes and representative 2.10.0 filenames are:

| Platform | Lane | Example wheel |
| --- | --- | --- |
| nvidia | `flagos-pypi-nvidia` | `torch_fl-2.10.0+cuda13.3-cp312-cp312-linux_x86_64.whl` |
| hygon | `flagos-pypi-hygon` | `torch_fl-2.10.0+dtk2604-cp310-cp310-linux_x86_64.whl` |
| enflame | `flagos-pypi-enflame` | `torch_fl-2.10.0+tops1.9.10-cp312-cp312-linux_x86_64.whl` |
| mthreads | `flagos-pypi-mthreads` | `torch_fl-2.10.0+musa5.2.0-cp310-cp310-linux_x86_64.whl` |
| thead | `flagos-pypi-thead` | `torch_fl-2.10.0+ppu2.0.0-cp312-cp312-linux_x86_64.whl` |
| ascend | `flagos-pypi-ascend` | `torch_fl-2.10.0+cann9.0.0-cp311-cp311-linux_aarch64.whl` |
| metax | `flagos-pypi-metax` | `torch_fl-2.10.0+maca3.8.1.3-cp312-cp312-linux_x86_64.whl` |

TsingMicro and BPU do not publish. Publication stays on vendor runners for a
measured network reason: a 2.35 GB artifact set took 59m47s to cross from the
GitHub artifact store to Nexus at about 0.13 MB/s, while the same route from a
runner in China measured about 13–23 MB/s. The build artifact hop is not the
bottleneck; the final package-index upload is.

Each upload defaults to a 900-second socket timeout, retries once on a bounded
transient failure, and is capped by a 20-minute workflow step. Pass
`--upload-timeout` for a deliberately different limit. The route probe selects
direct or proxy access to `resource.flagos.net`; set
`TORCH_FL_PROXY_ROUTE=direct` or `TORCH_FL_PROXY_ROUTE=proxy` when a runner
needs an explicit route. A dry run exercises lane and filename resolution
without credentials:

```bash
python3 .github/scripts/publish_wheels.py \
  --wheels /tmp/wheels --tag v2.10.0 --dry-run
```

Adding a publishing platform requires all four pieces to move together:

1. Add `pypi_lane` to `cmake/flagos_platforms.json`.
2. Add `.github/configs/<platform>.yml` with its build, acceptance and timeout contract.
3. Add the platform to the build, acceptance and publish matrices.
4. Supply an owned model workload, performance reference and real hardware evidence.

The release tests reject a lane that is present in only one of these places.

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

Candidate artifacts use the CI vendor lane name (`wheel-nvidia`, `wheel-hygon`,
`wheel-enflame`, `wheel-mthreads`, `wheel-thead`). The publisher resolves these
through the platform table and still accepts artifact directories named with
legacy build ids. Wheel versions keep their SDK segments unchanged.
