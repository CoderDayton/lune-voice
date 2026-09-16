# Release & build matrix

Lune Voice runs speech-to-text **in-process** through a native library. The
styling LLM is remote-only (OpenAI-compatible HTTP) — no native build step.

- ASR — [`whisper-rs`](https://github.com/tazz4843/whisper-rs) → bundled `whisper.cpp`

The `whisper-rs` `*-sys` crate **vendors and compiles the native C/C++ library
from source at `cargo build` time** (CMake + a C++ toolchain + `libclang` for
bindgen). It is vendored under `vendor/whisper-rs-sys/` via `[patch.crates-io]`,
with two modified files, both marked `LUNE FORK`: `build.rs` adds
`GGML_BACKEND_DL` support, and `src/lib.rs` adds a lint allow for the bindgen
output.

## Distribution model

Releases ship as finished Tauri bundles (`.deb`, `.rpm`, `.AppImage`, and a
Windows `-setup.exe`), so **end
users never compile** anything.

GPU support is **not** a separate artifact on Linux. The CUDA backend is a
`dlopen`-able plugin the user downloads from the app, so one CPU bundle serves
every Linux user and a machine with an NVIDIA driver opts in at runtime. See
[GPU_PLUGIN.md](GPU_PLUGIN.md) for how that works and how to publish the archive.

Linux and Windows are both built and tested; there is no macOS job yet. Each
OS-specific subsystem — audio capture, text delivery, the global hotkey, the
focused-window probe, the pill overlay — sits behind an interface with a half
per platform (see the platform-split table in `CLAUDE.md`), so CI runs the
**whole** workspace on `ubuntu-latest` and `windows-latest` alike. A bare local
`cgs build` auto-detects the build host; see Build commands.

Whisper models are **external assets**: GGML files are downloaded at runtime
into the application models directory (`~/.local/share/lune-voice/models/` on
Linux) and loaded from disk. They are **not** embedded in the executable or the
bundle.

## Release matrix

| Target | OS / arch | Backend | Cargo features | Status |
|---|---|---|---|---|
| Linux | Linux x86_64 | CPU (AVX2 baseline) | `whisper-cpp` | **Supported** |
| Linux GPU plugin | Linux x86_64 + NVIDIA | CUDA, downloaded at runtime | — (built by `cgs gpu-plugin`) | **Supported** |
| Windows CPU | Windows x86_64 | CPU (AVX2 baseline) | `whisper-cpp` | **Supported** (NSIS installer) |
| Windows GPU plugin | Windows x86_64 + NVIDIA | CUDA, downloaded at runtime | — (built by `cgs gpu-plugin-windows`) | **Wired; archive not published yet** |
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

`cgs build` **compiles only** — it produces the executable, not installers.
Shippable `.deb`/`.rpm`/`.AppImage`/`.tar.gz` come from `cgs package`, which
builds them in a container against an old glibc; a bundle made on the dev box
links that box's libraries and is not distributable. `--env BUNDLE=1` bundles
locally anyway, for testing the package layout itself.

It has two profiles. The default is a **native** build tuned to the build host
(fast, low memory); **release artifacts** must add `--env PORTABLE=1`. Backend
(CPU vs CUDA) auto-detects unless pinned with `--env BACKEND`.

```bash
# Local build (default): NATIVE — ggml targets only this machine's CPU + GPU
# arch. ~8x smaller CUDA compile, won't exhaust RAM. NOT portable to other HW.
cgs build

# Distributable release artifact (what CI uses, per target): PORTABLE —
# AVX2 CPU baseline + every common NVIDIA CUDA arch (heavy; needs lots of RAM).
cgs build --env PORTABLE=1 --env BACKEND=cpu
cgs build --env PORTABLE=1 --env BACKEND=cuda

# Installers, in a container (see Linux bundles below).
cgs package
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
RUSTFLAGS="-C target-cpu=x86-64-v3" \
  GGML_NATIVE=OFF GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON \
  bun tauri build --features "whisper-cuda"
```

For PORTABLE, `GGML_NATIVE=OFF` + the explicit AVX2/FMA/F16C flags force a
portable CPU baseline instead of ggml's default `-march=native` (which would
`SIGILL` on any user CPU older than the build host) and make ggml build all
common CUDA archs. The native default skips both — one CPU target, one GPU arch.

`RUSTFLAGS` covers the other half of the binary — the GGML flags do nothing for
the Rust code. `x86-64-v3` is the same AVX2 + FMA + F16C baseline the GGML flags
request; on a non-x86_64 host `cgs build` sets it empty instead.

Host-tuned codegen is opt-in per build, not pinned in `.cargo/config.toml`: a
committed `build.rustflags` would apply to a packager's plain `cargo build
--release` too, and that binary would `SIGILL` on any CPU older than the build
host. `cgs dev` and the native branch of `cgs build` export
`-C target-cpu=native` for themselves.

### Windows specifics

On Windows, run the equivalent in PowerShell from `desktop-ui\`:

```powershell
$env:RUSTFLAGS="-C target-cpu=x86-64-v3"
$env:GGML_NATIVE="OFF"; $env:GGML_AVX2="ON"; $env:GGML_FMA="ON"; $env:GGML_F16C="ON"
bun tauri build --features "whisper-cpp"     # CPU
bun tauri build --features "whisper-cuda"    # CUDA (needs CUDA Toolkit)
```

Tauri produces the `.msi`/`.exe` (NSIS) installers configured in
`desktop-ui/src-tauri/tauri.conf.json`.

From a Linux box, `cgs check-windows` type-checks the whole workspace —
`lune-ui` included — for `x86_64-pc-windows-gnu`, so the
`cfg(target_os = "windows")` halves of the platform split cannot rot between
pushes. It needs two things, both wired in by `.cargo/config.toml`: `zig` as the
C cross-compiler, and `binutils-mingw-w64-x86-64` for
`x86_64-w64-mingw32-windres`, which compiles `lune-ui`'s icon and manifest
resource with `zig cc` as its preprocessor. `lune-ui`'s build script reads
`desktop-ui/dist`, so build the frontend once on a fresh clone.

It is **check-only** — linking `x86_64-pc-windows-gnu` also needs a MinGW `gcc`
as the linker driver, so producing an actual Windows binary stays the CI job on
`windows-latest`.

### Linux bundles

`cgs package` (`scripts/release/package-all.sh`) builds `.deb`, `.AppImage`,
`.rpm` and a portable `.tar.gz` into `release-artifacts/`, in a single
`ubuntu:22.04` container (honors `$CONTAINER_ENGINE`). It builds the CPU backend
(`--features whisper-cpp`); GPU support reaches those bundles as the
downloadable CUDA plugin, not as a second artifact.

One image covers all three package formats because tauri writes RPMs with the
pure-Rust `rpm` crate — no `rpmbuild`, no Fedora. Ubuntu 22.04 is the right
single host because it has the oldest glibc of the distros targeted, so its
binary runs on the newer ones too. The `.rpm`'s `Requires` come from
`tauri.conf.json`, not from the build host.

The `.tar.gz` is the `.deb` payload repacked, not a second compile: the deb's
`usr/` tree is kept intact (tauri resolves resources as
`<prefix>/bin/lune-voice` -> `<prefix>/lib/Lune Voice`) with a top-level
`lune-voice` launcher and a `README.txt` covering the udev rule. The binary's
tauri bundle stamp is reset from `.deb` to "no bundle" (same length, and the
script refuses unless exactly one stamp is found), so the portable copy offers
a manual download instead of running `dpkg -i` through pkexec. Bundle file
names are normalized from `Lune Voice_...` to `lune-voice_...`, by
`package-all.sh` and by the release workflow alike.

| `--env TYPE=` | Produces |
|---|---|
| `all` (default) | bundles + tarball |
| `bundles` | `.deb`, `.AppImage`, `.rpm` |
| `tarball` | repacks the newest `.deb` already in `release-artifacts/` |

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

## CI

Two workflows cover the build.

**`.github/workflows/ci.yml`** gates every push and PR. Alongside fmt, clippy,
tests, DTO drift, the frontend and the dependency audits, a **`bundle`** job
builds the real Linux artifact on `ubuntu-22.04` — the same PORTABLE CPU build
`cgs build --env PORTABLE=1 --env BACKEND=cpu` produces — then asserts a `.deb`,
`.rpm` and `.AppImage` all landed and that the udev rule is actually inside the
`.deb`. Every other job compiles the whisper *stub*, so without this job a
feature-passthrough typo, a broken `resources` path or a missing icon passed CI
and failed at release time instead.

**`.github/workflows/release.yml`** runs on a `v*` tag. It first fails fast if
the tag disagrees with the version in `tauri.conf.json`, `Cargo.toml` and
`desktop-ui/package.json`, then builds the Linux bundles and the Windows
installer, writes checksums, and opens a **draft** GitHub release with them
attached. The Windows leg runs after the Linux one so only one job creates the
draft; it attaches to the release the Linux job made.

It stops at draft on purpose: the updater's private key never enters CI, so
`latest.json` does not exist yet. Finish by hand — see [In-app
updater](#in-app-updater) below. A release published without `latest.json` is a
release the updater cannot see.

> Status: **Linux CPU** and **Windows CPU** artifacts are both built. GPU
> reaches users as the downloadable CUDA plugin rather than a second bundle,
> and Windows has its own plugin build (`cgs gpu-plugin-windows`) whose archive
> is not published yet.
> `dmg` is not in `bundle.targets`: the macOS-Metal wiring remains to be
> implemented, and adding it means a `macos-14` leg in `release.yml` plus the
> feature work above.

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

## In-app updater

> **Currently off.** `UPDATES_ENABLED` in `desktop-ui/src-tauri/src/update.rs`
> is `false`. The endpoint in `tauri.conf.json` is a release asset on a private
> repository, and the updater plugin sends no credentials, so GitHub answers
> every check with a 404 — an update flow that always fails reads to a user as
> a broken app rather than an absent feature. **Check for Updates…** says so
> instead; the launch check stays silent, as it already does when offline.
>
> Everything below still describes how the flow works and how to publish for
> it. Flip the constant back once the release assets are anonymously
> reachable — the same condition the GPU plugin pin waits on
> ([GPU_PLUGIN.md](GPU_PLUGIN.md)). The signature verification is independent
> of all this and is still tested (`cargo test -p lune-ui --lib updater`).

The tray's **Check for Updates…** fetches `latest.json` from the latest GitHub
release, verifies the signature against the public key pinned in
`tauri.conf.json`, and installs: an AppImage is swapped in place, a `.deb`/`.rpm`
goes through `pkexec` (a graphical password prompt). Nothing downloads before
the user agrees, and the app restarts itself afterwards.

**The signing key is the trust root.** It lives at
`~/.tauri/lune-voice-updater.key` (mode 600), never in the repo. Anyone holding
it can push a signed update to every user; lose it and no existing install can
ever be updated again, because the public key is baked into shipped binaries.
Back it up somewhere you would not lose, and keep it out of `git` and CI logs.

Builds stay key-free on purpose. Turning on `createUpdaterArtifacts` would make
the bundler demand the private key on **every** `cgs build` — a pinned `pubkey`
without a matching private key aborts `tauri build` outright (*"A public key has
been found, but no private key"*), which would break the build for anyone who
does not hold it. So nothing is signed during bundling; `cgs updater-manifest`
packs the AppImage into the `.tar.gz` the plugin expects and signs all three
formats afterwards, in one place.

Per release. Bump the version in `tauri.conf.json`, `Cargo.toml` and
`desktop-ui/package.json`, then push the tag — `release.yml` builds the bundles
and opens the draft release. Sign and finish it locally:

```bash
git tag v0.2.0 && git push origin v0.2.0      # CI builds + drafts the release
rm -rf release-artifacts                      # no older release left behind
gh release download v0.2.0 -D release-artifacts
cgs updater-manifest --env TAG=v0.2.0         # signs, writes release-artifacts/latest.json
gh release upload v0.2.0 release-artifacts/latest.json
gh release edit v0.2.0 --draft=false
```

To build the bundles locally instead of on CI, `cgs package` replaces the tag
and download steps — but not the `rm -rf`. `cgs package` writes into the same
directory, and a previous release's artifacts left there are exactly what the
manifest's ambiguity check rejects. `cgs package` also builds Linux artifacts
only, so the resulting `latest.json` carries no `windows-x86_64` key.

`latest.json` must be uploaded to the release whose tag the endpoint resolves
to, or clients keep seeing the previous version. Bump `version` in
`tauri.conf.json` before building — the plugin compares against the running
app's own version, so a release built with a stale version is invisible.

The manifest carries one key per package format —
`linux-x86_64-{appimage,deb,rpm}` — because the plugin rejects bytes that do
not match how the app was installed, so a single URL cannot serve all three.
Windows has a single installer and so a single key, `windows-x86_64`.
Those are the keys the plugin looks up on its own: it tries
`{os}-{arch}-{installer}` for the format this install was packaged as and then
falls back to the bare `{os}-{arch}` key, so the app pins no target of its own.
The AppImage reaches clients as a `.tar.gz` (the plugin extracts it and looks
for the `.AppImage` inside); `cgs updater-manifest` builds and signs that
archive, and signs the deb and rpm, which the bundler would not sign even with
`createUpdaterArtifacts` on.
`cgs updater-manifest` refuses to run when the tag disagrees with the `version`
in `tauri.conf.json`, and it selects artifacts **by that version**, not by glob
order — glob order is version-ascending, so the oldest build used to win and the
manifest would advertise 0.2.0 while pointing at 0.1.0 files. Two errors come
out of that check:

- `no <kind> for <version> in <dir>, only: …` — the directory holds that format
  for some other version. Almost always a missing `rm -rf release-artifacts`.
- `more than one <kind> for <version>` — two builds of the same format and
  version, for example a `-rc.1` alongside the final. Remove the one you are not
  shipping.

Only the packed `.AppImage.tar.gz` and the portable archive may be absent.

### Testing the updater

Two layers, both local — nothing here needs a published release:

```bash
cargo test -p lune-ui --lib updater   # check + download + signature verification
cgs updater-test                      # the install path, for real
```

The unit tests stand a local HTTP endpoint in front of a fixture signed by a
throwaway key and assert three things: a correctly signed payload is accepted,
a payload with one flipped byte is refused, and a payload signed by a
*different valid key* is refused. The last two are the ones that matter — they
show verification is key-specific rather than merely well-formed.

`cgs updater-test` covers what those cannot: replacing the running app. It
builds the current version and a `99.0.0` AppImage, serves a signed manifest on
localhost, and prints the command to run the old one against it. Both bundles
are `--debug` deliberately: release builds refuse plain-HTTP endpoints, so
testing over HTTP needs no loosening of the shipped config, and the signature is
still verified against the pinned key. `LUNE_UPDATER_ENDPOINT` is what points a
build at the local manifest, and it is compiled out of release builds — a
shipped binary always talks to the endpoint in `tauri.conf.json`.

### How a found update is announced

It depends on who asked.

A check the user started — the tray's *Check for Updates…* or the button in
Settings → General — answers in a modal dialog, every time, including *"you are
on the latest version"* and every failure. A control that silently does nothing
is indistinguishable from a broken one.

A check started by **Check for updates on launch** (Settings → General, off by
default) posts a **desktop notification** instead, via
`tauri-plugin-notification` — libnotify on Linux, a WinRT toast on Windows,
`NSUserNotification` on macOS, with no permission handshake needed off mobile.
Twenty seconds into a session the user is working in another window, and a
dialog that seizes focus there is more disruptive than the update is urgent.
The notification announces and nothing more — *Lune Voice 1.4.0 is available* /
*You're running 1.3.2.* None of the desktop backends expose a click callback
(the plugin only wires notification actions up on Android and iOS), and
spelling out where to click instead would be worse: a notification is a glance,
not a manual. A launch check that finds *nothing*, or fails outright, stays
silent: "no network yet" is the normal state a second after login.

**Delivery is not confirmable.** Every desktop backend hands the notification to
a spawned task and returns `Ok` immediately, discarding the result, so a session
with no notification daemon looks identical to a delivered one. The dialog
fallback in `notify_update_available` therefore only catches the plugin
*rejecting* the call — it cannot catch a notification nobody saw. A user whose
session has no notification daemon gets no launch announcement at all, and finds
the update the next time they check by hand. Closing that hole properly needs a
durable in-app marker (a tray item that reads *Install 1.4.0…* until acted on)
rather than a better notification.

`cargo test -p lune-ui --lib notification_tests` covers what is assertable
without a daemon: the announcement text names the offered and installed versions
the right way round, the plugin accepts the builder chain, and each `CheckMode`
routes to the intended announcement. To see a real notification, turn the
setting on and start the `cgs updater-test` build against the local endpoint.

## Third-party notices

`whisper.cpp` and `ggml` are MIT-licensed. They are statically linked into the
Windows and default `cgs build` artifacts, and shipped as separate shared
objects wherever `GGML_BACKEND_DL` is enabled. The GPU plugin archive also
carries NVIDIA's redistributable CUDA runtime libraries. Keep
`THIRD_PARTY_NOTICES.md` in sync with the bundled upstream when its version
changes.

Every bundle ships two files:

- `THIRD_PARTY_NOTICES.md`, written by hand: native libraries, the CUDA runtime,
  models, fonts, the compiled-in data files and the vendored `whisper-rs-sys`
  fork.
- `THIRD_PARTY_LICENSES.md`, generated: the licence text of every Rust crate in
  the release build (cargo-about, `about.toml`) and every JavaScript package in
  the interface bundle (`desktop-ui/scripts/third-party-licenses.ts`).

Regenerate the second after any dependency change, and before `cgs package`:

```bash
cargo install cargo-about --locked --features cli --version 0.9.2   # once; same as release.yml
cgs licenses                                        # rewrite THIRD_PARTY_LICENSES.md
cgs licenses --env CHECK=1                          # verify only
```

The `build-linux` job in `release.yml` runs `--check` before building, so a tag
with a stale file fails instead of shipping wrong notices; `build-windows`
inherits that through `needs:`. cargo-about runs with `--fail`: a crate whose
licence it cannot determine stops the run, and needs a `clarify` entry in
`about.toml`.

**Nothing checks this outside a tag build.** `ci.yml` has no licence step and
`cgs package` does not call the script, so a dependency change can land on
`main`, and a local bundle can be built, with a stale `THIRD_PARTY_LICENSES.md`.
Run `cgs licenses --env CHECK=1` yourself after touching `Cargo.toml` or
`package.json`.

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
