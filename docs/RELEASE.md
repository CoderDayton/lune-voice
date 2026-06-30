# Release & build matrix

Lune Voice runs speech-to-text and the styling LLM **in-process** through native
libraries:

- ASR — [`whisper-rs`](https://github.com/tazz4843/whisper-rs) → bundled `whisper.cpp`
- LLM — [`llama-cpp-2`](https://github.com/utilityai/llama-cpp-rs) → bundled `llama.cpp` (via `llama-cpp-sys-2`)

Both `*-sys` crates **vendor and compile the native C/C++ library from source at
`cargo build` time** (CMake + a C++ toolchain + `libclang` for bindgen). The GPU
backend is selected at **build time** by a Cargo feature; there is **no runtime
backend autodetection and no external inference server** (no `llama-server`).

## Distribution model

Releases are **prebuilt per platform + backend** and shipped as finished Tauri
bundles. **End users never compile** anything. Each artifact is built on a runner
that has the matching toolchain (and, for GPU artifacts, the matching SDK), with
the backend feature set **explicitly** — never auto-detected from the build host.

Models are **external assets**: GGUF / GGML files are downloaded at runtime into
the application models directory (`~/.local/share/lune-voice/models/` on Linux)
and loaded from disk. They are **not** embedded in the executable or the bundle.

## Release matrix

| Target | OS / arch | Backend | Cargo features | Status |
|---|---|---|---|---|
| Linux CPU | Linux x86_64 | CPU (AVX2 baseline) | `whisper-cpp llama-cpp` | **Supported** |
| Linux CUDA | Linux x86_64 + NVIDIA | CUDA | `whisper-cuda llama-cuda` | **Supported** |
| Windows CPU | Windows x86_64 | CPU (AVX2 baseline) | `whisper-cpp llama-cpp` | **Supported** |
| Windows CUDA | Windows x86_64 + NVIDIA | CUDA | `whisper-cuda llama-cuda` | **Supported** |
| macOS Metal | macOS arm64 (Apple Silicon) | Metal | `whisper-metal llama-metal` | **Planned** (see below) |

The `whisper-cuda`/`llama-cuda` features forward to `whisper-rs/cuda` and
`llama-cpp-2/cuda`; the CPU features just compile the libraries without a GPU
backend. The Cargo feature graph is cross-platform — the same `--features` flags
work on Linux and Windows; only the build toolchain differs per OS.

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

CPU and CUDA builds are wired into `cgs build` (explicit, no autodetection):

```bash
# Linux/Windows CPU (portable AVX2 baseline) — the default:
cgs build                       # == BACKEND=cpu
cgs build --env BACKEND=cpu

# Linux/Windows CUDA (NVIDIA; build host needs the CUDA Toolkit):
cgs build --env BACKEND=cuda
```

Under the hood that runs, from `desktop-ui/`:

```bash
# CPU
GGML_NATIVE=OFF GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON \
  bun tauri build --features "whisper-cpp llama-cpp"

# CUDA
GGML_NATIVE=OFF GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON \
  bun tauri build --features "whisper-cuda llama-cuda"
```

`GGML_NATIVE=OFF` + the explicit AVX2/FMA/F16C flags force a portable CPU
baseline instead of ggml's default `-march=native`, which would `SIGILL` on any
user CPU older than the build host.

### Windows specifics

On Windows, run the equivalent in PowerShell from `desktop-ui\`:

```powershell
$env:GGML_NATIVE="OFF"; $env:GGML_AVX2="ON"; $env:GGML_FMA="ON"; $env:GGML_F16C="ON"
bun tauri build --features "whisper-cpp llama-cpp"     # CPU
bun tauri build --features "whisper-cuda llama-cuda"   # CUDA (needs CUDA Toolkit)
```

Tauri produces the `.msi`/`.exe` (NSIS) installers configured in
`desktop-ui/src-tauri/tauri.conf.json`.

### Linux bundles

`scripts/release/package-all.sh` builds `.deb` / `.AppImage` / `.rpm` via
containerized cross-distro builds (honors `$CONTAINER_ENGINE`). Pass the backend
through `BACKEND` as above when invoking the production build inside it.

## macOS Metal (planned, not yet wired)

To add the macOS Apple-Silicon artifact:

1. Add `metal` passthrough features mirroring the CUDA ones:
   - `crates/lune-asr/Cargo.toml`: `whisper-metal = ["whisper-cpp", "whisper-rs/metal"]`
   - `crates/lune-llm/Cargo.toml`: `llama-metal = ["llama-cpp", "llama-cpp-2/metal"]`
   - `crates/lune-app` + `desktop-ui/src-tauri`: `whisper-metal`/`llama-metal`
     passthrough + a `metal` umbrella (mirroring `gpu`).
2. Gate the runtime `use_gpu` decision on `cfg!(feature = "llama-metal")` /
   `cfg!(feature = "whisper-metal")` alongside the existing CUDA `cfg!`s
   (`desktop-ui/src-tauri/src/runtime.rs`, `crates/lune-asr/src/backend/registry.rs`).
3. Build on a macOS arm64 runner:
   `bun tauri build --features "whisper-metal llama-metal"` → `.dmg`.

`whisper-rs` and `llama-cpp-2` both expose a `metal` feature, so no upstream work
is required — only the passthrough wiring above. (`coreml` for whisper and
`vulkan`/`rocm` for either are available too, but are out of scope unless an
explicit product requirement.)

## CI release matrix (planned)

A GitHub Actions release workflow (e.g. `.github/workflows/release.yml`,
triggered on a `v*` tag) should fan out across a matrix that pins each artifact
to its backend. Sketch:

```yaml
strategy:
  matrix:
    include:
      - { name: linux-cpu,    os: ubuntu-22.04,  features: "whisper-cpp llama-cpp",   cuda: false }
      - { name: linux-cuda,   os: ubuntu-22.04,  features: "whisper-cuda llama-cuda", cuda: true  }
      - { name: windows-cpu,  os: windows-2022,  features: "whisper-cpp llama-cpp",   cuda: false }
      - { name: windows-cuda, os: windows-2022,  features: "whisper-cuda llama-cuda", cuda: true  }
      - { name: macos-metal,  os: macos-14,      features: "whisper-metal llama-metal", cuda: false }
```

Each leg: install Rust + CMake + LLVM (+ Bun + platform WebView deps), install the
CUDA Toolkit on `cuda: true` legs, then run the per-target build above and upload
the bundle from `desktop-ui/src-tauri/target/release/bundle/` (and `release-artifacts/`
for the Linux containerized bundles) as a release asset.

> Status: the build commands and Cargo features for **Linux/Windows × CPU/CUDA**
> are in place today. The macOS-Metal feature wiring and the CI workflow above are
> documented here and remain to be implemented (they require macOS/Windows CI
> runners to validate).

## Third-party notices

`whisper.cpp`, `llama.cpp`, and `ggml` are MIT-licensed and statically linked
into the shipped binary. Keep `THIRD_PARTY_NOTICES.md` in sync with the bundled
upstreams when their versions change.
