# Release & build matrix

Lune Voice runs speech-to-text **in-process** through a native library. The
styling LLM is remote-only (OpenAI-compatible HTTP) — no native build step.

- ASR — [`whisper-rs`](https://github.com/tazz4843/whisper-rs) → bundled `whisper.cpp`

The `whisper-rs` `*-sys` crate **vendors and compiles the native C/C++ library
from source at `cargo build` time** (CMake + a C++ toolchain + `libclang` for
bindgen). It is vendored under `vendor/whisper-rs-sys/` via `[patch.crates-io]`,
with a single modified file (`build.rs`) that adds `GGML_BACKEND_DL` support.

## Distribution model

Releases are **prebuilt per platform** and shipped as finished Tauri bundles.
**End users never compile** anything.

GPU support is **not** a separate artifact on Linux. The CUDA backend is a
`dlopen`-able plugin the user downloads from the app, so one CPU bundle serves
every Linux user and a machine with an NVIDIA driver opts in at runtime. See
[GPU_PLUGIN.md](GPU_PLUGIN.md) for how that works and how to publish the archive.

Windows still builds a pinned CPU and CUDA artifact per target; release runners
pass `--env BACKEND` so each is reproducible. (A bare local `cgs build` instead
auto-detects the build host; see Build commands.)

Whisper models are **external assets**: GGML files are downloaded at runtime
into the application models directory (`~/.local/share/lune-voice/models/` on
Linux) and loaded from disk. They are **not** embedded in the executable or the
bundle.

## Release matrix

| Target | OS / arch | Backend | Cargo features | Status |
|---|---|---|---|---|
| Linux | Linux x86_64 | CPU (AVX2 baseline) | `whisper-cpp` | **Supported** |
| Linux GPU plugin | Linux x86_64 + NVIDIA | CUDA, downloaded at runtime | — (built by `cgs gpu-plugin`) | **Supported** |
| Windows CPU | Windows x86_64 | CPU (AVX2 baseline) | `whisper-cpp` | **Supported** |
| Windows CUDA | Windows x86_64 + NVIDIA | CUDA | `whisper-cuda` | **Supported** |
| macOS Metal | macOS arm64 (Apple Silicon) | Metal | `whisper-metal` | **Planned** (see below) |

The `whisper-cuda` feature forwards to `whisper-rs/cuda`; the CPU feature just
compiles the library without a GPU backend. The Cargo feature graph is
cross-platform — the same `--features` flag works on Linux and Windows; only
the build toolchain differs per OS.

The Linux GPU plugin is not a Cargo feature at all. It is built separately with
`cgs gpu-plugin` and published as a standalone `.tar.gz` on the GitHub release;
the application downloads and verifies it on request.

## Toolchain per runner

All targets need: **Rust** (stable), **CMake**, a **C/C++ compiler**, and
**LLVM/Clang including `libclang`** (for bindgen), plus **Bun** for the Tauri UI
build and the platform's Tauri/WebView prerequisites.

- **Linux**: gcc/clang, `libwebkit2gtk-4.1-dev`, `libclang` (`LIBCLANG_PATH` may
  need setting, e.g. `/lib/x86_64-linux-gnu`). CUDA target also needs the **CUDA
  Toolkit** (`nvcc`).
- **Windows**: MSVC build tools (or clang), CMake, LLVM (`libclang`). CUDA target
  also needs the **CUDA Toolkit**.
- **macOS** (planned): Xcode command-line tools provide Metal; CMake + LLVM.

## Build commands

`cgs build` has two profiles. The default is a **native** build tuned to the
build host (fast, low memory); **release artifacts** must add `--env PORTABLE=1`.
Backend (CPU vs CUDA) auto-detects unless pinned with `--env BACKEND`.

```bash
# Local build (default): NATIVE — ggml targets only this machine's CPU + GPU
# arch. ~8x smaller CUDA compile, won't exhaust RAM. NOT portable to other HW.
cgs build

# Distributable release artifact (what CI uses, per target): PORTABLE —
# AVX2 CPU baseline + every common NVIDIA CUDA arch (heavy; needs lots of RAM).
cgs build --env PORTABLE=1 --env BACKEND=cpu
cgs build --env PORTABLE=1 --env BACKEND=cuda
```

> **Why two profiles:** with `GGML_NATIVE=OFF` (PORTABLE) ggml compiles ~8 CUDA
> architectures; at high `-j` each `nvcc`/`cicc` job needs several GB and the
> build can exhaust RAM and crash (`cicc` SIGSEGV). Native builds one arch and
> avoids this. For PORTABLE on a many-core host, cap jobs with `CARGO_BUILD_JOBS`.

Under the hood that runs, from `desktop-ui/`:

```bash
# Native (default): tuned to this machine's CPU + GPU
GGML_NATIVE=ON bun tauri build --features "whisper-cuda"

# Portable (PORTABLE=1): portable CPU baseline + all common CUDA archs
GGML_NATIVE=OFF GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON \
  bun tauri build --features "whisper-cuda"
```

For PORTABLE, `GGML_NATIVE=OFF` + the explicit AVX2/FMA/F16C flags force a
portable CPU baseline instead of ggml's default `-march=native` (which would
`SIGILL` on any user CPU older than the build host) and make ggml build all
common CUDA archs. The native default skips both — one CPU target, one GPU arch.

### Windows specifics

On Windows, run the equivalent in PowerShell from `desktop-ui\`:

```powershell
$env:GGML_NATIVE="OFF"; $env:GGML_AVX2="ON"; $env:GGML_FMA="ON"; $env:GGML_F16C="ON"
bun tauri build --features "whisper-cpp"     # CPU
bun tauri build --features "whisper-cuda"    # CUDA (needs CUDA Toolkit)
```

Tauri produces the `.msi`/`.exe` (NSIS) installers configured in
`desktop-ui/src-tauri/tauri.conf.json`.

### Linux bundles

`scripts/release/package-all.sh` builds `.deb` / `.AppImage` / `.rpm` via
containerized cross-distro builds (honors `$CONTAINER_ENGINE`). Pass the backend
through `BACKEND` as above when invoking the production build inside it.

## macOS Metal (planned, not yet wired)

To add the macOS Apple-Silicon artifact:

1. Add a `metal` passthrough feature mirroring the CUDA one:
   - `crates/lune-asr/Cargo.toml`: `whisper-metal = ["whisper-cpp", "whisper-rs/metal"]`
   - `crates/lune-voice` + `desktop-ui/src-tauri`: `whisper-metal`
     passthrough + a `metal` umbrella (mirroring `gpu`).
2. Gate the runtime `use_gpu` decision on `cfg!(feature = "whisper-metal")`
   alongside the existing CUDA `cfg!`s
   (`crates/lune-asr/src/backend/registry.rs`).
3. Build on a macOS arm64 runner:
   `bun tauri build --features "whisper-metal"` → `.dmg`.

`whisper-rs` exposes a `metal` feature, so no upstream work is required — only
the passthrough wiring above. (`coreml` for whisper and `vulkan`/`rocm` are
available too, but are out of scope unless an explicit product requirement.)

## CI release matrix (planned)

A GitHub Actions release workflow (e.g. `.github/workflows/release.yml`,
triggered on a `v*` tag) should fan out across a matrix that pins each artifact
to its backend. Sketch:

```yaml
strategy:
  matrix:
    include:
      - { name: linux-cpu,    os: ubuntu-22.04,  features: "whisper-cpp",    cuda: false }
      - { name: linux-cuda,   os: ubuntu-22.04,  features: "whisper-cuda",   cuda: true  }
      - { name: windows-cpu,  os: windows-2022,  features: "whisper-cpp",    cuda: false }
      - { name: windows-cuda, os: windows-2022,  features: "whisper-cuda",   cuda: true  }
      - { name: macos-metal,  os: macos-14,      features: "whisper-metal",  cuda: false }
```

Each leg: install Rust + CMake + LLVM (+ Bun + platform WebView deps), install the
CUDA Toolkit on `cuda: true` legs, then run the per-target build above and upload
the bundle from `desktop-ui/src-tauri/target/release/bundle/` (and `release-artifacts/`
for the Linux containerized bundles) as a release asset.

> Status: the build commands and Cargo features for **Linux/Windows × CPU/CUDA**
> are in place today. The macOS-Metal feature wiring and the CI workflow above are
> documented here and remain to be implemented (they require macOS/Windows CI
> runners to validate).

## GPU plugin

Built and packaged separately from the application bundle:

```bash
cgs gpu-plugin        # -> release-artifacts/lune-voice-gpu-cuda13-linux-x86_64.tar.gz + .sha256
cgs stage-runtime     # copy ggml shared objects + CPU plugins next to a built binary
```

Upload both files to the GitHub release, then record the digest and byte size in
`CUDA_LINUX_X86_64` (`crates/lune-persist/src/gpu_plugin.rs`). Until that digest
is filled in, the in-app install refuses rather than fetching something it
cannot verify. Full details in [GPU_PLUGIN.md](GPU_PLUGIN.md).

## Third-party notices

`whisper.cpp` and `ggml` are MIT-licensed. They are statically linked into the
Windows and default `cgs build` artifacts, and shipped as separate shared
objects wherever `GGML_BACKEND_DL` is enabled. The GPU plugin archive also
carries NVIDIA's redistributable CUDA runtime libraries. Keep
`THIRD_PARTY_NOTICES.md` in sync with the bundled upstream when its version
changes.

Lune Voice itself is GPL-3.0-or-later; `cargo deny check licenses` gates the
dependency tree against the allow-list in `deny.toml`.
