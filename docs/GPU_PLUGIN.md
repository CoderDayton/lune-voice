# GPU backend plugin

GPU acceleration ships as a **downloadable plugin**, not as a separate build of
the application. One CPU binary runs everywhere; a user with an NVIDIA card opts
in and the CUDA backend arrives at runtime.

This document covers why it works that way, how to build and publish the
archive, and what to check when it misbehaves.

## Why a plugin

The obvious design is two release artifacts, one linked against CUDA and one
not. That has two costs: the CUDA build refuses to start on a machine with no
NVIDIA driver (a hard `DT_NEEDED` on `libcudart` is resolved before `main`
runs), and every user has to pick the right download.

The equally obvious fix, shipping the CUDA libraries and pointing
`LD_LIBRARY_PATH` at them from inside the process, does not work. glibc reads
`LD_LIBRARY_PATH` once at startup into a cached search list, so calling `setenv`
later changes nothing for subsequent `dlopen` calls.

What does work is ggml's own plugin mode:

- The app is built with **`GGML_BACKEND_DL=ON`**, which turns ggml's compute
  backends into `dlopen`-able shared objects instead of libraries linked into
  the executable. The binary then carries no reference to CUDA at all.
- Each plugin is linked with **`RUNPATH=$ORIGIN`**, so it resolves its own
  dependencies from its own directory without any environment variable.

`GGML_BACKEND_DL` requires `BUILD_SHARED_LIBS=ON` and is incompatible with
`GGML_NATIVE`, so plugin builds pass `GGML_NATIVE=OFF` and use
`GGML_CPU_ALL_VARIANTS=ON` to get per-microarchitecture CPU plugins dispatched
at runtime.

## Two plugin directories

| Directory | Holds | Shipped how |
|---|---|---|
| `<exe dir>/backends/` | CPU backends | Bundled with the app. Not optional: without them there is no compute backend at all. |
| `<data dir>/backends/` | `libggml-cuda.so` + NVIDIA runtime | Downloaded on request. Absent until the user opts in. |

On Linux the data dir is `~/.local/share/lune-voice/`.

Both are scanned at startup by `lune_asr::gpu_backend::load_plugins`, which
**must run before the first `WhisperContext` is created** — ggml picks a device
from whatever is in the backend registry at context-creation time, so a plugin
loaded later has no effect until the next launch. `desktop-ui/src-tauri/src/lib.rs`
calls it before the Tauri builder for that reason.

Loading is idempotent via a `OnceLock`: ggml's registry is process-global and
has no unload, so loading twice would register duplicate devices.

## What ships in the archive

`lune-voice-gpu-cuda13-linux-x86_64.tar.gz` — roughly 471 MB compressed:

| File | Size | Why |
|---|---|---|
| `libggml-cuda.so` | 106 MB | The backend itself, `sm_75/86/89/120a` + `compute_75` PTX |
| `libcublasLt.so.13` | 503 MB | NVIDIA. See below |
| `libcublas.so.13` | 54 MB | NVIDIA |
| `libcudart.so.13` | 0.8 MB | NVIDIA |

`libcuda.so.1` is deliberately **not** included. It ships with the driver, not
the toolkit, and is not redistributable — which is why a missing driver is the
one failure a download cannot fix.

### On the size

cuBLASLt is roughly three quarters of the unpacked payload (503 MB of 664 MB)
and cannot be reduced. Measured, not assumed:

- `libggml-cuda.so` references **zero** cuBLASLt symbols directly, but
  `libcublas.so.13` declares it `DT_NEEDED` and imports 76 symbols from it, so
  it can be neither dropped nor stubbed.
- Static linking with `--gc-sections` produces **541 MB** for the ten cuBLAS
  entry points ggml actually uses — the dispatcher reaches every kernel object,
  so roughly 3% is prunable.
- `nvprune` only rewrites relocatable device code, not final shared objects.
- Recompressing with zstd-19 saves about 6%; fatbins are already compressed.
- CUDA 12 is not smaller. CUDA 12.4 is the leanest 12.x at 551 MB for the cuBLAS
  pair against CUDA 13.1's 557 MB, and 12.9 is far worse at 749 MB for cuBLASLt
  alone.

The one real argument for CUDA 12 is hardware coverage, not size: CUDA 13
dropped Maxwell, Pascal, and Volta, so GTX 10-series cards get no GPU option.

## Architectures

`cgs gpu-plugin` defaults to `CUDA_ARCHS="75-real;86-real;89-real;120-real;75-virtual"`:
native SASS for the four consumer generations CUDA 13 supports (Turing, Ampere,
Ada, Blackwell), plus `compute_75` PTX as a floor. PTX JITs **forward** only, so
the oldest virtual arch covers everything else — A100, H100, datacenter
Blackwell — at the cost of a one-off JIT compile on first run.

Two things to know:

- ggml rewrites `12X` into the architecture-specific `12Xa`, because Blackwell's
  FP4 tensor-core instructions are not forward-compatible. A request for `120`
  is therefore satisfied by `sm_120a`, and the build's verification step allows
  that suffix.
- CMake **caches** `CMAKE_CUDA_ARCHITECTURES`. Reusing a build directory that was
  configured for a narrower list silently keeps producing that narrower plugin —
  an easy way to ship an `sm_89`-only archive by accident. The script
  reconfigures from scratch whenever the list changes, and verifies the built
  binary with `cuobjdump` rather than trusting the configure step.

## Building and publishing

```bash
cgs gpu-plugin                              # default arch set
cgs gpu-plugin --env CUDA_ARCHS="89-real"   # narrow, for a local test
```

The script builds `libggml-cuda.so`, stages it with the three NVIDIA
redistributables, strips it, asserts `RUNPATH=$ORIGIN` is present, verifies the
architectures against the binary, then writes the tarball and its `.sha256` into
`release-artifacts/`.

To publish:

1. Upload both files to the GitHub release named in `CUDA_LINUX_X86_64.url`.
2. Record the digest and byte size in `CUDA_LINUX_X86_64`
   (`crates/lune-persist/src/gpu_plugin.rs`).

Until the digest is filled in it stays at the `UNPUBLISHED` sentinel and
`install` refuses with an actionable message rather than downloading something
it cannot verify.

## Runtime states

`lune_asr::gpu_backend::state` returns exactly one of these, and each maps to a
single thing the UI can say or offer:

| State | Meaning | UI |
|---|---|---|
| `Unsupported` | No native backend compiled in (stub build) | Section hidden entirely |
| `DriverMissing` | No NVIDIA driver. A download would not help | Explains, offers nothing |
| `NotInstalled` | Driver present, plugin not downloaded | The offer, with size |
| `Ready` | A CUDA device is registered | Names the device |
| `Failed` | Plugin present, no device appeared | Says restart, or driver too old |

Driver detection `dlopen`s `libcuda.so.1` and lets the dynamic linker search its
normal paths. Scanning `/usr/local/cuda` would both miss distro packages that
put the driver elsewhere and find a *toolkit* with no driver behind it.

`Ready` is decided by a registered device whose name starts with `CUDA`, which
is the right test for both shapes of build: the downloaded plugin, and a
statically linked `whisper-cuda` binary that has no plugin file on disk yet
needs no download either.

The `Failed` distinction matters. If the plugin was not on disk when the
registry was built, the cause is simply that ggml cannot load it mid-process, so
the message says to restart. Only when it *was* present at load time and still
registered nothing does the message blame the driver.

IPC surface: `gpu_backend_status` and `install_gpu_backend`, the latter
streaming progress over a Tauri `Channel`.

## Download safety

`lune_persist::gpu_plugin::install` streams with `reqwest`, hashing as it goes,
and:

- refuses outright when the digest is `UNPUBLISHED`;
- verifies SHA-256 against the pinned value before unpacking anything;
- extracts only an allowlist of flat filenames (`libggml-cuda.so`,
  `libcudart.so.*`, `libcublas.so.*`, `libcublasLt.so.*`), so a tampered archive
  cannot write outside the plugin directory via `../` or absolute paths.

## Troubleshooting

**The plugin loads but no CUDA device appears.** Almost always the driver is
older than the CUDA runtime. CUDA 13 needs r580 or newer on Linux. Check with
`nvidia-smi`.

**`ldd` shows the CUDA libraries resolving from `/usr/local/cuda` instead of the
plugin directory.** Expected on a developer machine: `LD_LIBRARY_PATH` outranks
`RUNPATH`. Test the way a user's machine behaves:

```bash
env -u LD_LIBRARY_PATH ldd <dir>/libggml-cuda.so
```

All three NVIDIA libraries should resolve to `<dir>`. `libggml-base.so.0`
showing as not found is normal there — it ships with the app, not the archive.

**Verifying what a built plugin actually contains:**

```bash
cuobjdump -lelf libggml-cuda.so | grep -oP 'sm_\K[0-9]+[a-z]*' | sort -u
cuobjdump -lptx libggml-cuda.so | grep -oP 'sm_\K[0-9]+[a-z]*' | sort -u
```

## Known limitation

`cgs build` still produces a **static** artifact. The `GGML_BACKEND_DL` layout
needs the ggml shared objects and CPU plugins staged next to the executable so
`$ORIGIN` resolves, and Tauri's `resources` mechanism stages into the resource
directory rather than the bin directory. Until that is wired and a bundle is
verified end to end, `cgs build` keeps producing the known-good static build.

`cgs dev` and `cgs stage-runtime` do exercise the plugin path.

## Running a dev build on the GPU

`cgs dev` sets `GGML_BACKEND_DL=ON`, so a CUDA dev build emits the CUDA backend
as a plugin rather than linking it in — and a plugin ggml never finds is a
plugin it never uses. Build it, then stage it:

```bash
cgs dev --env BACKEND=cuda   # once, to produce libggml-cuda.so
cgs stage-runtime            # copies it into target/debug/backends/
```

`stage-runtime` reports whether CUDA was among the plugins it staged. This is
only needed for `cgs dev`; `cgs build` links CUDA in statically and needs no
staging.
