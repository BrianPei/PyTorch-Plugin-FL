# Iluvatar CI image

CI reuses the operator-published baseline
`harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20261016-iluvatar-cpu-dev`,
pinned by its manifest digest:
`sha256:bc1b3c0beb22b301ec501896ea1c77903879526fb7303358669321d609458023`.
The runner label is `ts-v150-cicd-pytorch`.

## Official CPU core and isolation

`/opt/torch-fl-iluvatar-venv` has `include-system-site-packages=false`, official
`torch==2.10.0+cpu`, FlagTree `0.6.2a3+iluvatar3.6` (including Triton 3.6),
FlagGems `5.4.0`, and build/test dependencies. Inherited PYTHONPATH is cleared
and user-site packages are disabled. Base-image vLLM and vendor Python packages
remain outside the venv. No torch_fl wheel is baked into the image.

The runtime added by upstream PR #593 uses the CoreX C ABI with the official
CPU torch headers, libraries and C++ ABI. It needs neither vendor libtorch nor
a replacement torch core. After importing torch_fl, `torch.version.cuda` stays
`None`; the compatibility shim exposes the device operations needed by
FlagGems and Triton through `torch.cuda`.

The published image predates #593 and contains unused assets under
`/opt/torch-fl-iluvatar-libtorch`. The current hook clears its legacy
`FLAGOS_VENDOR_TORCH_LIB` default, persists the empty value through GITHUB_ENV,
and never copies those assets into a wheel. The checkout's verifier is used
instead of the older copy baked into the image. These changes allow reuse of
the existing digest without a Harbor push.

CI compares the preinstalled FlagTree/Gems versions with checkout pins and
job overrides. Matching versions are reused; a changed component is installed
inside the disposable job container. FlagTree and its Triton provider are
replaced together. The checkout's constraints protect CPU torch, and `pip check`
rejects incompatible cohorts. SDK, Python and CPU torch baseline changes need
a rebuilt and revalidated image.

## Recipe for the next image refresh

This directory keeps the image recipe, FlagTree RECORD-based copier and
verification tool together so future image changes are reviewable. The current
Dockerfile omits the obsolete vendor libtorch staging step; it is the recipe
for the next refresh, not an exact reproduction of the already-published image.
It retains the pinned CoreX 4.5 development SDK that supplies the required
IX/ixblas and ixdnn headers.

```bash
DOCKER_BUILDKIT=0 docker build -f docker/iluvatar/Dockerfile \
  -t harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20261020-iluvatar-cpu-runtime-dev .
```

The default CPU-wheel mirror is the SJTU mirror also supported by the shared
CI helper. Set `CPU_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` to use
the official index directly. Ordinary packages use `PIP_INDEX_URL`.

Build-time `verify.py` checks CPU torch, ABI, isolation, package pins and SDK
headers without devices. Once a wheel is installed, `verify.py --device` also
checks the mapped CPU core libraries, native ELF dependencies, BI-V150 Triton
target (corex, arch 71, warp 64), Triton JIT and flagos addition. The shared
CI manifest runs this installed-wheel audit using `--runtime` before its tests.

See [the bring-up guide](../../docs/vendors/iluvatar/installation.md) for host
mounts and the outstanding hardware acceptance checks. The active configuration
is [`.github/configs/iluvatar.yml`](../../.github/configs/iluvatar.yml).
