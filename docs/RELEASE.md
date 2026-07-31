# Release & build matrix

Lune Voice runs speech-to-text **in-process** through a native library. The
styling LLM is remote-only (OpenAI-compatible HTTP) — no native build step.

- ASR — [`whisper-rs`](https://github.com/tazz4843/whisper-rs) → bundled `whisper.cpp`

The `whisper-rs` `*-sys` crate **vendors and compiles the native C/C++ library
from source at `cargo build` time** (CMake + a C++ toolchain + `libclang` for
bindgen). It is vendored under `vendor/whisper-rs-sys/` via `[patch.crates-io]`,
with a single modified file (`build.rs`) that adds `GGML_BACKEND_DL` support.

## Distribution model

Releases ship as finished Tauri bundles (`.deb`, `.rpm`, `.AppImage`), so **end
users never compile** anything.

GPU support is **not** a separate artifact on Linux. The CUDA backend is a
`dlopen`-able plugin the user downloads from the app, so one CPU bundle serves
every Linux user and a machine with an NVIDIA driver opts in at runtime. See
[GPU_PLUGIN.md](GPU_PLUGIN.md) for how that works and how to publish the archive.

Only Linux is built and tested. Audio capture and text delivery are Linux-only
(ALSA/PipeWire, `uinput`, IBus, evdev hotkeys), which is why CI's Windows job
excludes `lune-audio`, `lune-voice` and `lune-ui`, and why there is no macOS job
at all. A bare local `cgs build` auto-detects the build host; see Build commands.

Whisper models are **external assets**: GGML files are downloaded at runtime
into the application models directory (`~/.local/share/lune-voice/models/` on
Linux) and loaded from disk. They are **not** embedded in the executable or the
bundle.

## Release matrix

| Target | OS / arch | Backend | Cargo features | Status |
|---|---|---|---|---|
| Linux | Linux x86_64 | CPU (AVX2 baseline) | `whisper-cpp` | **Supported** |
| Linux GPU plugin | Linux x86_64 + NVIDIA | CUDA, downloaded at runtime | — (built by `cgs gpu-plugin`) | **Supported** |
| Windows CPU | Windows x86_64 | CPU (AVX2 baseline) | `whisper-cpp` | **Not built** |
| Windows CUDA | Windows x86_64 + NVIDIA | CUDA | `whisper-cuda` | **Not built** |
| macOS Metal | macOS arm64 (Apple Silicon) | Metal | `whisper-metal` | **Planned** (see below) |

The `whisper-cuda` feature forwards to `whisper-rs/cuda`; the CPU feature just
compiles the library without a GPU backend. The Cargo feature graph is
cross-platform — the same `--features` flag works on Linux and Windows — but that
feature graph is the only cross-platform part; the crates a desktop app needs on
top of it are not.

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

> Not buildable end to end today (see the matrix note above). Kept for when the
> Linux-only capture and delivery layers grow a Windows path.

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
the bundle from `target/release/bundle/` (and `release-artifacts/` for the Linux
containerized bundles) as a release asset.

> Status: only the **Linux** build is exercised today. The Windows CPU/CUDA Cargo
> features exist, but the Linux-only audio and delivery crates do not compile
> there, so no Windows artifact is produced. The macOS-Metal wiring and the
> release workflow above remain to be implemented.

## GPU plugin

Built and packaged separately from the application bundle:

```bash
cgs gpu-plugin        # -> release-artifacts/lune-voice-gpu-cuda13-linux-x86_64.tar.gz + .sha256
cgs stage-runtime     # copy ggml shared objects + backend plugins next to a built binary
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

## Rollback

Rolling back is reinstalling the previous package. There is no server side and
no coordinated migration, so the only real question is what the newer version
did to on-disk state before you rolled back.

### What a rollback has to survive

| State | Rollback-safe? | Why |
|---|---|---|
| `settings.toml` | Mostly | Unknown keys are ignored on load, so a newer file loads on an older build. Fields the older build does not know are **dropped on its next save**. |
| `lune.db` | Only within the same schema version | Diesel records applied migrations. An older binary will not downgrade the schema on its own. |
| Models | Yes | Plain files, keyed by directory name. |
| GPU plugin | Yes | Independent of the app version; delete the directory to revert to CPU. |
| Keyring secrets | Yes | Slot names are stable and versionless. |

### Procedure

1. **Back up state first** — this is the step that makes the rollback reversible:

   ```sh
   cp -r ~/.local/share/lune-voice ~/.local/share/lune-voice.bak
   cp -r ~/.config/lune-voice      ~/.config/lune-voice.bak
   ```

2. **Reinstall the previous package.**

   ```sh
   sudo apt install --allow-downgrades ./lune-voice_<old>_amd64.deb   # deb
   sudo dnf downgrade ./lune-voice-<old>.x86_64.rpm                   # rpm
   ```

   For an AppImage, just run the older file.

3. **Check whether the schema moved.** Compare the migrations the old binary
   embeds against what the DB has applied:

   ```sh
   sqlite3 ~/.local/share/lune-voice/lune.db \
     "SELECT version FROM __diesel_schema_migrations ORDER BY version;"
   ```

   Versions newer than anything in the old build's `crates/lune-persist/migrations/`
   mean the DB is ahead of the binary. Additive migrations (every one shipped so
   far) are tolerated — the old code simply ignores the new columns. If the old
   build fails to start, revert the extra migrations explicitly rather than
   deleting the database:

   ```sh
   diesel migration revert --migration-dir crates/lune-persist/migrations
   ```

   Each migration's `down.sql` is exercised by
   `migrations_revert_and_reapply_cleanly` in `crates/lune-persist/src/db/mod.rs`,
   so reverting is a tested path, not a hope.

4. **Verify** dictation end-to-end before declaring it done:

   ```sh
   cgs bench --env TYPE=transcribe --env WAV=fixtures/<clip>.wav
   ```

### If the rollback itself goes wrong

Restore the backups from step 1 and reinstall the newer package:

```sh
rm -rf ~/.local/share/lune-voice ~/.config/lune-voice
mv ~/.local/share/lune-voice.bak ~/.local/share/lune-voice
mv ~/.config/lune-voice.bak      ~/.config/lune-voice
```

### Release-side rule

Because an older build silently drops settings fields it does not know, a
release that **removes or renames** a `settings.toml` field is not
rollback-safe. Migrate such fields in a release that still reads both spellings,
and only drop the old one a release later.
