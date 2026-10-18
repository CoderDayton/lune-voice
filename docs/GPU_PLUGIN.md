# GPU backend plugin

GPU acceleration ships as a **downloadable plugin**, not as a separate build of
the app. One CPU binary runs everywhere. Someone with an NVIDIA card opts in, and
the CUDA backend is downloaded and loaded at runtime.

This page covers why it works that way, how to build and publish the archive,
and what to check when it breaks.

## Why a plugin

The first idea is two release builds, one linked against CUDA and one not. That
costs two things: the CUDA build won't start on a machine with no NVIDIA driver
(a hard `DT_NEEDED` on `libcudart` is resolved before `main` runs), and every
user has to pick the right download.

The second idea, shipping the CUDA libraries and pointing `LD_LIBRARY_PATH` at
them from inside the process, doesn't work. glibc reads `LD_LIBRARY_PATH` once
at startup and caches the search list, so a later `setenv` has no effect on
`dlopen`.

ggml's own plugin mode does work:

- The app is built with **`GGML_BACKEND_DL=ON`**, which turns ggml's compute
  backends into `dlopen`-able shared objects instead of libraries linked into
  the executable. The binary then carries no reference to CUDA at all.
- Each plugin is linked with **`RUNPATH=$ORIGIN`**, so it resolves its own
  dependencies from its own directory without any environment variable.

`GGML_BACKEND_DL` requires `BUILD_SHARED_LIBS=ON` and is incompatible with
`GGML_NATIVE`, so plugin builds pass `GGML_NATIVE=OFF` and use
`GGML_CPU_ALL_VARIANTS=ON` to get per-microarchitecture CPU plugins dispatched
at runtime. Release bundles ship two of them, AVX2 and SSE4.2.

## Two plugin directories

| Directory | Holds | Shipped how |
|---|---|---|
| `<exe dir>/backends/` | CPU backends | Bundled with the app. Not optional: without them there is no compute backend at all. |
| `<data dir>/backends/` | `libggml-cuda.so` + NVIDIA runtime | Downloaded on request. Absent until the user opts in. |

On Linux the data dir is `~/.local/share/lune-voice/`. An installed Linux bundle
keeps its CPU backends in Tauri's resource directory instead,
`<prefix>/lib/LuneVoice/backends/`, which `bundled_plugin_dir` falls back to
when there is no `backends/` beside the executable. How release builds produce
and stage them: [Build commands](RELEASE.md#build-commands).

Both are scanned at startup by `lune_asr::gpu_backend::load_plugins`, which
**must run before the first `WhisperContext` is created**. ggml picks a device
from whatever is in the backend registry when the context is created, so a
plugin loaded later does nothing until the next launch. That is why
`desktop-ui/src-tauri/src/lib.rs` calls it before the Tauri builder.

A `OnceLock` makes loading run only once. ggml's registry is global to the
process and has no unload, so loading twice would register every device twice.

## What ships in the archive

`lune-voice-gpu-cuda13-linux-x86_64.tar.gz`, about 471 MB compressed:

| File | Size | Why |
|---|---|---|
| `libggml-cuda.so` | 106 MB | The backend itself, `sm_75/86/89/120a` + `compute_75` PTX |
| `libcublasLt.so.13` | 503 MB | NVIDIA. See below |
| `libcublas.so.13` | 54 MB | NVIDIA |
| `libcudart.so.13` | 0.8 MB | NVIDIA |
| `ggml-version` | 7 B | The ggml version the plugin was built against |

`libcuda.so.1` is **not** included on purpose. It comes with the driver, not the
toolkit, and can't be redistributed. That is why a missing driver is the one
problem a download can't fix.

The Windows archive, `lune-voice-gpu-cuda13-windows-x86_64.tar.gz`, holds the
same four libraries plus `ggml-version` under their Windows names
(`ggml-cuda.dll`, `cublasLt64_13.dll`, `cublas64_13.dll`, `cudart64_13.dll`).
It leaves out the driver (`nvcuda.dll`) for the same reason. It is a `.tar.gz`,
not a `.zip`, so both platforms share one verified extractor. Only the file
names differ, and the allowlist in `gpu_plugin.rs` picks the set by platform.

### How a plugin finds its CUDA runtime

Each platform does this differently:

- **Linux** bakes `RUNPATH=$ORIGIN` into `libggml-cuda.so` at link time, so it
  finds the other libraries in its own directory. `LD_LIBRARY_PATH` can't be
  used, because glibc caches the search list at startup, before the app could
  set it.
- **Windows** has no `$ORIGIN`. ggml loads plugins with a plain `LoadLibraryW`
  (`ggml-backend-reg.cpp`), and a DLL loaded by full path looks for *its*
  imports in the executable's directory, System32 and `PATH`, never in its own
  folder. So the app loads `cudart`, `cublasLt` and `cublas` by full path
  before ggml scans the directory (`preload_sibling_libraries` in
  `crates/lune-asr/src/gpu_backend.rs`). Windows then matches the import to the
  module that is already loaded. Changing the search order with
  `SetDefaultDllDirectories` instead would affect every other `LoadLibrary` in
  the process, WebView2's included.

### On the size

cuBLASLt is about three quarters of the unpacked size (503 MB of 664 MB), and it
can't be made smaller. These were all measured:

- `libggml-cuda.so` uses **no** cuBLASLt symbols directly, but
  `libcublas.so.13` lists it as `DT_NEEDED` and imports 76 symbols from it, so
  it can't be dropped or stubbed out.
- Static linking with `--gc-sections` gives **541 MB** for the ten cuBLAS entry
  points ggml uses. The dispatcher reaches every kernel object, so only about
  3% can be pruned.
- `nvprune` only rewrites relocatable device code, not final shared objects.
- Recompressing with zstd-19 saves about 6%; fatbins are already compressed.
- CUDA 12 is not smaller. CUDA 12.4 is the leanest 12.x at 551 MB for the cuBLAS
  pair against CUDA 13.1's 557 MB, and 12.9 is far worse at 749 MB for cuBLASLt
  alone.

The one reason to prefer CUDA 12 is hardware support, not size. CUDA 13 dropped
Maxwell, Pascal and Volta, so GTX 10-series cards get no GPU option.

## Architectures

`cgs gpu-plugin` defaults to `CUDA_ARCHS="75-real;86-real;89-real;120-real;75-virtual"`:
native SASS for the four consumer generations CUDA 13 supports (Turing, Ampere,
Ada, Blackwell), plus `compute_75` PTX as a fallback. PTX only compiles
**forward** to newer GPUs, so the oldest virtual arch covers everything else
(A100, H100, datacenter Blackwell), at the cost of a one-time JIT compile on
first run.

Two things to know:

- ggml rewrites `12X` into the architecture-specific `12Xa`, because Blackwell's
  FP4 tensor-core instructions are not forward-compatible. A request for `120`
  is therefore satisfied by `sm_120a`, and the build's verification step allows
  that suffix.
- CMake **caches** `CMAKE_CUDA_ARCHITECTURES`. A build directory configured for
  a shorter list quietly keeps building that shorter plugin, which makes it
  easy to ship an `sm_89`-only archive by accident. The script reconfigures
  from scratch whenever the list changes, and checks the built binary with
  `cuobjdump` instead of trusting the configure step.

## Building and publishing

```bash
cgs gpu-plugin                              # Linux, default arch set
cgs gpu-plugin --env CUDA_ARCHS="89-real"   # narrow, for a local test
cgs gpu-plugin-windows                      # Windows; must run ON Windows
```

They run `scripts/release/gpu-plugin-linux.sh` and
`scripts/release/gpu-plugin-windows.sh`. The Linux script builds
`libggml-cuda.so`, copies in the three NVIDIA libraries, strips the result,
checks that `RUNPATH=$ORIGIN` is set, checks the architectures in the binary,
writes the `ggml-version` stamp (ggml's `MAJOR.MINOR.PATCH` from
`ggml/CMakeLists.txt`), and writes the tarball and its `.sha256` into
`release-artifacts/`.

A plugin only works with the ggml it was built against. At load time ggml only
checks its backend API version, which doesn't change when the structs it shares
with a plugin do. A plugin from a different ggml registers its device and then
crashes the app. So the app loads the downloaded plugin only when its stamp
matches `ggml_version()`, and otherwise decodes on the CPU. Any whisper.cpp
upgrade that changes ggml's version needs a rebuilt plugin under a new release
tag.

`cgs gpu-plugin-windows` works the same way. Run it from Git Bash, in a shell
where MSVC and the CUDA toolkit are on `PATH` and `CUDA_PATH` is set. CUDA 13
keeps its DLLs in `bin\x64`, and the script looks there before `bin`. There is
no `$ORIGIN` to check, so it checks the staged files against the app's
allowlist instead. One stray file would make the install fail for every user.
**CUDA device code can't be cross-compiled**: nvcc builds for the host
toolchain, so this archive can only be made on a Windows machine with the CUDA
toolkit.

The **GPU plugin** workflow (`.github/workflows/gpu-plugin.yml`) builds both
archives on GitHub's runners and attaches them to a draft release. You start it
by hand, since a new plugin is only needed when ggml's version changes. It
refuses a tag whose release is already published:

```bash
gh workflow run gpu-plugin.yml -f tag=gpu-cuda13-v1
```

The Linux archive builds on Ubuntu 22.04, the same base as the app bundles, so
it needs the same minimum glibc as the app. The run summary lists each
archive's byte size and digest for step 2 below.

To publish, for either platform:

1. Upload both files to the GitHub release named in the release constant's
   `url` (`CUDA_LINUX_X86_64` or `CUDA_WINDOWS_X86_64`).
2. Record the digest and byte size in that constant
   (`crates/lune-persist/src/gpu_plugin.rs`).
3. Update `no_release_is_published_while_the_assets_are_unreachable` in that
   file. It fails on purpose once a digest is pinned, so it tells you what to
   change instead of quietly passing.

Until then the digest stays `UNPUBLISHED`, and `install` refuses with a message
that says what to do, instead of downloading something it can't verify.

**Neither platform is published yet**, so the app doesn't offer the download
anywhere. The repository is private, and GitHub answers the anonymous request
`install` makes with a 404. Pinning a digest now would only turn a hidden button
into a broken one. The measured Linux digest and byte size are kept in
`CUDA_LINUX_X86_64_SHA256` and `CUDA_LINUX_X86_64_BYTES` next to the constant.
Once the files can be downloaded without logging in (a public repo, or another
host), turning them on is a two-line edit. The pinned digest is what makes an
untrusted host safe, so it doesn't have to be GitHub.

Building the plugin locally with `cgs gpu-plugin` and `cgs stage-runtime` still
works, because that path never downloads anything.

## Runtime states

`lune_asr::gpu_backend::state` returns one of these, and each one maps to one
thing the UI shows or offers:

| State | Meaning | UI |
|---|---|---|
| `Unsupported` | No native backend compiled in (stub build), or ggml linked statically without CUDA, which never loads plugins | Section hidden entirely |
| `DriverMissing` | No NVIDIA driver. A download would not help | Explains, offers nothing |
| `NotInstalled` | Driver present, plugin not downloaded or built against another ggml | The offer, with size |
| `Ready` | A CUDA device is registered | Names the device |
| `NeedsRestart` | Plugin installed after the registry was built | Asks for a restart |
| `Failed` | Plugin present at startup, no device appeared | Blames the driver, with the reason |

To find the driver, the app `dlopen`s `libcuda.so.1` and lets the dynamic
linker search its usual paths. Scanning `/usr/local/cuda` instead would miss
distro packages that put the driver elsewhere, and would find a *toolkit* with
no driver behind it.

`Ready` means a registered device's name starts with `CUDA`. That test works
for both kinds of build: the downloaded plugin, and a statically linked
`whisper-cuda` binary, which has no plugin file but doesn't need a download
either.

`NeedsRestart` and `Failed` both mean "installed, but no device", and the
difference matters. If the plugin wasn't on disk when the registry was built,
ggml just can't load it mid-run, so the state is `NeedsRestart` with no
`reason`, because nothing is wrong. Only when the plugin *was* there at startup
and still registered nothing is the state `Failed`, and its `reason` points at
the driver.

The registry is built once at startup, so `NeedsRestart` only follows an
install made by the running app. A fresh launch reports `Ready` or `Failed`.

IPC commands: `gpu_backend_status` and `install_gpu_backend`. The second
streams progress over a Tauri `Channel`.

## Download safety

`lune_persist::gpu_plugin::install` streams with `reqwest`, hashing as it goes,
and:

- refuses outright when the digest is `UNPUBLISHED`;
- verifies SHA-256 against the pinned value before unpacking anything;
- extracts only an allowlist of flat filenames (`libggml-cuda.so`,
  `libcudart.so.*`, `libcublas.so.*`, `libcublasLt.so.*`, `ggml-version`), so a tampered archive
  cannot write outside the plugin directory via `../` or absolute paths;
- records a per-file SHA-256 manifest (`.integrity.json`) before publishing any
  of them.

The manifest carries the pin through to load time. The archive digest proves
what was *downloaded*. Before each `dlopen`, `lune_asr::gpu_backend` hashes
every recorded file again, because these libraries run their constructors
inside a process that holds an exclusive grab on every keyboard. If the manifest is missing,
unreadable or doesn't match, the directory is skipped and reported as not
installed, which offers the download again.

## Troubleshooting

**GPU is offered again, and the log says the backend "was built against another
ggml".** The installed plugin is older than the app's ggml, or has no
`ggml-version` stamp. The app skipped it and decodes on the CPU. Install a
plugin built from the same whisper.cpp tree as the app.

**The plugin loads but no CUDA device appears.** The driver is almost always
older than the CUDA runtime. CUDA 13 needs r580 or newer on Linux. Check with
`nvidia-smi`.

**`ldd` shows the CUDA libraries coming from `/usr/local/cuda` instead of the
plugin directory.** That is normal on a developer machine, because
`LD_LIBRARY_PATH` wins over `RUNPATH`. To see what a user's machine does:

```bash
env -u LD_LIBRARY_PATH ldd <dir>/libggml-cuda.so
```

All three NVIDIA libraries should resolve to `<dir>`. `libggml-base.so.0` showing
as not found is fine here, since it ships with the app, not the archive.

**To check what a built plugin contains:**

```bash
cuobjdump -lelf libggml-cuda.so | grep -oP 'sm_\K[0-9]+[a-z]*' | sort -u
cuobjdump -lptx libggml-cuda.so | grep -oP 'sm_\K[0-9]+[a-z]*' | sort -u
```

## Running a dev build on the GPU

`cgs dev` sets `GGML_BACKEND_DL=ON`, so a CUDA dev build produces the CUDA
backend as a plugin instead of linking it in. ggml only uses a plugin it can
find, so build it, then stage it:

```bash
cgs dev --env BACKEND=cuda   # once, to produce libggml-cuda.so
cgs stage-runtime            # copies it into target/debug/backends/
```

`stage-runtime` tells you whether CUDA was among the plugins it copied. This is
only needed for `cgs dev`. The native `cgs build` links CUDA in statically and
needs no staging.
