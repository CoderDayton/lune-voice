# GPU backend plugin

GPU acceleration is a **downloadable plugin**, not a separate build. One CPU
binary runs everywhere; anyone with a Vulkan 1.2 GPU (NVIDIA, AMD, Intel) opts
in and ggml's Vulkan backend is downloaded and loaded at runtime.

## Why a plugin

A GPU-linked build won't start on a machine without the GPU runtime (a hard
`DT_NEEDED` resolves before `main`), and users would have to pick the right
download. Setting `LD_LIBRARY_PATH` from inside the process doesn't help either:
glibc caches the search list at startup.

So the app uses ggml's plugin mode:

- Built with **`GGML_BACKEND_DL=ON`**: compute backends are `dlopen`-able
  shared objects, and the binary has no GPU dependency at all.
- Each plugin carries **`RUNPATH=$ORIGIN`** and resolves its own dependencies.

`GGML_BACKEND_DL` needs `BUILD_SHARED_LIBS=ON` and rules out `GGML_NATIVE`, so
plugin builds pass `GGML_NATIVE=OFF` and `GGML_CPU_ALL_VARIANTS=ON`. Release
bundles ship two CPU plugins, AVX2 and SSE4.2.

## Why Vulkan

One plugin covers every vendor, and the runtime (`libvulkan.so.1` /
`vulkan-1.dll`) comes with the graphics driver, so nothing else is
redistributed. Measured on an RTX 4090 (median decode over 10 clips, warm):

| Model | CPU | CUDA | Vulkan |
|---|---|---|---|
| Large v3 Turbo | 6092 ms | 288 ms | 313 ms |
| Small | 2575 ms | 355 ms | 420 ms |

Vulkan is within 10–20% of CUDA at about 17 MB instead of about 490 MB. The
first decode after install can stall a few seconds while the driver compiles
shaders; the driver caches them after that.

## Plugin directories

| Directory | Holds | Shipped how |
|---|---|---|
| `<exe dir>/backends/` | CPU backends | Bundled. Required. |
| `<data dir>/backends/vulkan/` | `libggml-vulkan.so` + `ggml-version` | Downloaded on request. |

Linux data dir: `~/.local/share/lune-voice/`. An installed Linux bundle keeps
its CPU backends in `<prefix>/lib/LuneVoice/backends/`, which
`bundled_plugin_dir` falls back to. See [Build commands](RELEASE.md#build-commands).

The download gets its own `vulkan/` folder so nothing else that lands in
`backends/` gets scanned along with it. At startup the app deletes the files of
the earlier CUDA plugin from `backends/` itself
(`lune_persist::gpu_plugin::remove_legacy_plugin`): exact names only, nothing
in subfolders.

`lune_asr::gpu_backend::load_plugins` scans both, and **must run before the
first `WhisperContext`**: ggml picks a device from the registry at context
creation. `desktop-ui/src-tauri/src/lib.rs` calls it before the Tauri builder.
A `OnceLock` keeps it to one load; ggml's registry has no unload, so a second
load would register every device twice.

## The archive

`lune-voice-gpu-vulkan-linux-x86_64.tar.gz` and
`lune-voice-gpu-vulkan-windows-x86_64.tar.gz`. Each holds two files:

| File | Why |
|---|---|
| `libggml-vulkan.so` / `ggml-vulkan.dll` | The backend, shaders compiled in |
| `ggml-version` | The ggml version it was built against |

The Linux plugin may only need `libggml-base.so.0`, `libvulkan.so.1` and the
C/C++ runtime; the build script fails on anything else.

## Building and publishing

```bash
cgs gpu-plugin    # runs scripts/release/gpu-plugin-vulkan.sh for this OS
```

Needs cmake and the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
(for `glslc` and the SPIR-V headers), with `VULKAN_SDK` set: source the SDK's
`setup-env.sh` on Linux. On Windows, run it from Git Bash in an MSVC shell.

The script builds `ggml-vulkan`, strips it, checks its `NEEDED` list and
`RUNPATH=$ORIGIN` (Linux), writes the `ggml-version` stamp from
`ggml/CMakeLists.txt`, refuses any extra file, and writes the tarball and
`.sha256` into `release-artifacts/`.

A plugin only works with the ggml it was built against. ggml's load check only
compares its backend API version, so a mismatched plugin registers and then
crashes. The app loads the download only when its stamp matches
`ggml_version()`, otherwise it decodes on the CPU. A whisper.cpp upgrade that
moves ggml's version needs a rebuilt plugin under a new tag.

The **GPU plugin** workflow (`.github/workflows/gpu-plugin.yml`) builds both
archives and attaches them to a draft release. It installs the Vulkan SDK from
LunarG, pinned by sha256, and refuses a tag that is already published:

```bash
gh workflow run gpu-plugin.yml -f tag=gpu-v1
```

Linux builds on Ubuntu 22.04, the same base as the app bundles. The run summary
lists each archive's size and digest.

To publish:

1. Publish the draft release named in the constant's `url`
   (`VULKAN_LINUX_X86_64` / `VULKAN_WINDOWS_X86_64`).
2. Put the digest and byte size in that constant
   (`crates/lune-persist/src/gpu_plugin.rs`).

While the digest is `UNPUBLISHED`, the app hides the download and `install`
refuses.

## Runtime states

`lune_asr::gpu_backend::state`:

| State | Meaning | UI |
|---|---|---|
| `Unsupported` | Stub build, ggml linked statically, or no discrete GPU (integrated only) | Section hidden |
| `DriverMissing` | No Vulkan loader or no device | Explains, offers nothing |
| `DriverOutdated` | A discrete GPU whose driver predates Vulkan 1.2 | Asks for a driver update |
| `NotInstalled` | Discrete Vulkan 1.2 GPU, plugin missing or built against another ggml | The offer, with size |
| `Ready` | A discrete GPU registered (`Vulkan*`, or `CUDA*` in a dev build) | Names the device |
| `NeedsRestart` | Plugin installed after the registry was built | Asks for a restart |
| `Failed` | Plugin there at startup, but no discrete GPU registered | The reason: integrated only, or no Vulkan 1.2 |

The driver check asks the system Vulkan loader for its physical devices
(`crates/lune-asr/src/vulkan_probe.rs`), off the UI thread. A found GPU is
kept for the session; any other answer is asked again after 10 s, so a driver
installed while the app is open shows up without a restart. On Windows the
loader is taken from System32 only. Integrated GPUs and software renderers such
as llvmpipe don't count.

Decoding runs on the first discrete GPU ggml registered. whisper.cpp numbers
integrated and discrete GPUs together, so the app passes that GPU's index
instead of letting whisper.cpp take whichever comes first, which on a laptop
is often the integrated one.

`NeedsRestart` has no `reason`: nothing is wrong, ggml just can't load mid-run.
A fresh launch reports `Ready` or `Failed`.

IPC: `gpu_backend_status` and `install_gpu_backend` (progress over a Tauri
`Channel`).

## Download safety

`lune_persist::gpu_plugin::install` streams with `reqwest`, hashing as it goes:

- refuses when the digest is `UNPUBLISHED`;
- checks SHA-256 against the pin before unpacking;
- extracts only the two allowlisted flat names, so `../` and absolute paths
  can't escape;
- writes a per-file SHA-256 manifest (`.integrity.json`) and publishes the
  plugin file last.

Before each `dlopen`, `lune_asr::gpu_backend` re-hashes every file in the
manifest, because the plugin runs inside a process that holds a keyboard grab.
A missing or mismatched manifest means "not installed", which offers the
download again.

## Troubleshooting

**Download is offered again and the log says "built against another ggml".**
Stale plugin. Install one built from the app's whisper.cpp tree.

**Plugin loads but no Vulkan device appears.** Check `vulkaninfo --summary`.
The driver needs Vulkan 1.2, and a software-only device (llvmpipe) doesn't
count.

**Check what a plugin links:**

```bash
readelf -d libggml-vulkan.so | grep -E 'NEEDED|RUNPATH'
```

## Running a dev build on the GPU

`cgs dev` and `cgs build` pick the GPU backend themselves: Vulkan when `glslc`
is on `PATH` and `pkg-config` finds the Vulkan headers (source the Vulkan SDK's
`setup-env.sh`), else CUDA when `nvcc` works, else CPU. They print the pick.
Force one with `--env BACKEND=cpu|cuda|vulkan`.

`cgs dev` builds the GPU backend as a plugin, so run `cgs stage-runtime` once to
copy it into `target/debug/backends/`. The native `cgs build` links it
statically and needs no staging. Only `BACKEND=vulkan` runs the same backend
users get; CUDA stays a dev-only option.
