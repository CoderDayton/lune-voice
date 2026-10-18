# Release & build matrix

Lune Voice runs speech-to-text **inside the app** through a native library. The
styling LLM is remote only (OpenAI-compatible HTTP), so it has no native build
step.

- ASR: [`whisper-rs`](https://github.com/tazz4843/whisper-rs) → bundled `whisper.cpp`

The `whisper-rs` `*-sys` crate **compiles the native C/C++ library from source
during `cargo build`** (CMake, a C++ toolchain, and `libclang` for bindgen). It
is vendored under `vendor/whisper-rs-sys/` through `[patch.crates-io]`, with
whisper.cpp v1.9.4 inside it. Three files are modified, and every change is
marked `LUNE FORK`:

- `build.rs` adds `GGML_BACKEND_DL` support, refreshes the compiled copy of
  whisper.cpp whenever the vendored tree changes, and reads the 1.9 version
  declaration.
- `src/lib.rs` adds a lint allow for the bindgen output.
- `whisper.cpp/src/whisper.cpp` makes language auto-detection honour
  `audio_ctx` and reuse its encode for the first window, so `language = "auto"`
  costs no second encoder pass.

To upgrade whisper.cpp, copy the same subset of upstream over
`vendor/whisper-rs-sys/whisper.cpp/`, reapply the `LUNE FORK` hunk in
`src/whisper.cpp`, and run the `whisper_lang_detect` test.

## Distribution model

Releases ship as finished Tauri bundles (`.deb`, `.rpm`, `.AppImage`, and a
Windows `-setup.exe`), so **users never compile** anything.

GPU support is **not** a separate download of the app. The CUDA backend is a
`dlopen`-able plugin the user downloads from inside the app, so one CPU bundle
serves everyone, and a machine with an NVIDIA driver opts in at runtime. See
[GPU_PLUGIN.md](GPU_PLUGIN.md) for how that works and how to publish the archive.

Linux and Windows are both built and tested. There is no macOS build. Each
OS-specific subsystem (audio capture, text delivery, the global hotkey, the
focused-window check, the pill overlay) sits behind one interface with a Linux
half and a Windows half (see the platform table in `CLAUDE.md`), so CI runs the
**whole** workspace on both `ubuntu-latest` and `windows-latest`. A plain local
`cgs build` detects what the machine has; see Build commands.

Whisper models are **not** in the executable or the bundle. The GGML files are
downloaded at runtime into the models directory
(`~/.local/share/lune-voice/models/` on Linux) and loaded from disk.

## Release matrix

| Target | OS / arch | Backend | Cargo features | Status |
|---|---|---|---|---|
| Linux | Linux x86_64 | CPU (any x86-64-v2; best CPU variant picked at runtime) | `whisper-cpp` | **Supported** |
| Linux GPU plugin | Linux x86_64 + NVIDIA | CUDA, downloaded at runtime | none (built by `cgs gpu-plugin`) | **Built; archive not published yet** |
| Windows CPU | Windows x86_64 | CPU (any x86-64-v2; best CPU variant picked at runtime) | `whisper-cpp` | **Supported** (NSIS installer) |
| Windows GPU plugin | Windows x86_64 + NVIDIA | CUDA, downloaded at runtime | none (built by `cgs gpu-plugin-windows`) | **Built; archive not published yet** |

The `whisper-cuda` feature passes through to `whisper-rs/cuda`. The CPU feature
compiles the library with no GPU backend. The same `--features` flags work on
Linux and Windows, but only the features are shared: the crates the desktop app
needs on top of them differ by platform.

The GPU plugin is not a Cargo feature at all. It is built separately with
`cgs gpu-plugin` and published as its own `.tar.gz` on a GitHub release. The
app downloads and verifies it when the user asks.

## Toolchain per runner

All targets need: **Rust** (stable), **CMake**, a **C/C++ compiler**, and
**LLVM/Clang including `libclang`** (for bindgen), plus **Bun** for the Tauri UI
build and the platform's Tauri/WebView prerequisites.

- **Linux**: gcc/clang, `libwebkit2gtk-4.1-dev`, `libclang` (`LIBCLANG_PATH` may
  need setting, e.g. `/lib/x86_64-linux-gnu`). CUDA target also needs the **CUDA
  Toolkit** (`nvcc`).
- **Windows**: MSVC build tools (or clang), CMake, LLVM (`libclang`). CUDA target
  also needs the **CUDA Toolkit**.

## Build commands

`cgs build` **only compiles**. It makes the executable, not installers. The
`.deb`/`.rpm`/`.AppImage`/`.tar.gz` you ship come from `cgs package`, which
builds them in a container against an old glibc. A bundle made on your own
machine links your machine's libraries and won't run everywhere. `--env
BUNDLE=1` bundles locally anyway, for testing the package layout.

`cgs build` has two profiles. The default is a **native** build tuned to the
machine building it (fast, low memory). **Release builds** use `--env
PORTABLE=1`. Native builds pick CPU or CUDA by themselves unless you set `--env
BACKEND`. PORTABLE is CPU only, because the GPU ships as the downloadable
plugin.

```bash
# Local build (default): NATIVE. Static; ggml targets only this machine's CPU
# and GPU. The CUDA compile is ~8x smaller and won't run out of RAM. Does NOT
# run on other hardware.
cgs build

# Distributable release artifact (what CI and `cgs package` build): PORTABLE.
cgs build --env PORTABLE=1

# Installers, in a container (see Linux bundles below).
cgs package
```

Native runs, from `desktop-ui/`:

```bash
GGML_NATIVE=ON bun tauri build --features "whisper-cuda"
```

PORTABLE runs `scripts/release/build-bundles.sh`. Every distributable build
uses it: `cgs build`, `cgs package`, CI and the release workflow.

- ggml is built as shared libraries with `GGML_BACKEND_DL=ON`, and the bundle
  ships two CPU backend plugins: `haswell` (AVX2 + FMA + F16C + BMI2) and
  `sse42` for CPUs without AVX2. At startup ggml loads the best one the CPU
  supports. ggml can't build only some variants, so `GGML_CPU_ALL_VARIANTS=ON`
  compiles all of them and the script keeps those two (`CPU_VARIANTS`). This
  same layout is what lets the downloaded CUDA plugin load (docs/GPU_PLUGIN.md).
- The Rust code has one baseline, `RUSTFLAGS=-C target-cpu=x86-64-v2`
  (SSE4.2 + POPCNT), set by the script. That is the oldest CPU a release runs
  on.
- It builds twice: `tauri build --no-bundle` compiles everything, the ggml
  libraries and CPU plugins are staged into `desktop-ui/src-tauri/runtime/`,
  and a second `tauri build` with `tauri.runtime.conf.json` bundles that
  directory as resources. Only the app crate recompiles.

Once installed, the libraries sit in Tauri's resource directory. On Linux that
is `<prefix>/lib/LuneVoice/`, which the executable's `RUNPATH` points to. On
Windows it is the install directory, since Windows finds DLLs next to the
executable. The CPU plugins are in `backends/` below it. To run a PORTABLE
binary from the build tree, copy them there first:
`cgs stage-runtime --env PROFILE=release`.

Tuning for the build machine is turned on per build, not set in
`.cargo/config.toml`. A committed `build.rustflags` would also apply to a
packager's plain `cargo build --release`, and that binary would crash with
`SIGILL` on any CPU older than the one that built it. `cgs dev` and the native
`cgs build` set `-C target-cpu=native` for themselves.

### Windows specifics

On Windows, run the same script from Git Bash at the repository root, in a
shell where MSVC, CMake and LLVM are on `PATH` (`LIBCLANG_PATH` set):

```bash
bash scripts/release/build-bundles.sh
```

Tauri builds the NSIS `-setup.exe` installer set up in
`desktop-ui/src-tauri/tauri.conf.json`.

On Linux, `cgs check-windows` type-checks the whole workspace (`lune-ui`
included) for `x86_64-pc-windows-gnu`, so the Windows code can't quietly break
between pushes. It needs two things, both set up in `.cargo/config.toml`: `zig`
as the C cross-compiler, and `binutils-mingw-w64-x86-64` for
`x86_64-w64-mingw32-windres`, which compiles `lune-ui`'s icon and manifest
resource with `zig cc` as its preprocessor. `lune-ui`'s build script reads
`desktop-ui/dist`, so build the frontend once on a fresh clone.

It **only checks**. Linking for `x86_64-pc-windows-gnu` would also need MinGW
`gcc` as the linker, so real Windows binaries are built by the CI job on
`windows-latest`.

### Linux bundles

`cgs package` (`scripts/release/package-all.sh`) builds `.deb`, `.AppImage`,
`.rpm` and a portable `.tar.gz` into `release-artifacts/`, in a single
`ubuntu:22.04` container (it respects `$CONTAINER_ENGINE`). It builds the CPU
backend (`--features whisper-cpp`). GPU support reaches these bundles as the
downloadable CUDA plugin, not as a second build.

One image covers all three package formats because tauri writes RPMs with the
pure-Rust `rpm` crate, so there is no `rpmbuild` and no Fedora. Ubuntu 22.04 has
the oldest glibc of the distros we target, so a binary built there runs on the
newer ones too. The `.rpm`'s `Requires` come from `tauri.conf.json`, not from
the build machine.

The `.tar.gz` is the `.deb`'s contents repacked, not a second compile. The
deb's `usr/` tree is kept as is (tauri finds resources as
`<prefix>/bin/lune-voice` -> `<prefix>/lib/LuneVoice`), with a `lune-voice`
launcher at the top and a `README.txt` about the udev rule. The binary's tauri
bundle stamp is changed from `.deb` to "no bundle" (same length, and the script
stops unless it finds exactly one stamp). That way the portable copy offers a
manual download instead of running `dpkg -i` through pkexec. Both
`package-all.sh` and the release workflow rename bundle files from
`LuneVoice_...` to `lune-voice_...`.

| `--env TYPE=` | Produces |
|---|---|
| `all` (default) | bundles + tarball |
| `bundles` | `.deb`, `.AppImage`, `.rpm` |
| `tarball` | repacks the newest `.deb` already in `release-artifacts/` |

## CI

Two workflows cover the build.

**`.github/workflows/ci.yml`** checks every push and PR. Besides fmt, clippy,
tests, DTO drift, the frontend and the dependency audits, a **`bundle`** job
builds the real Linux bundles on `ubuntu-22.04` (the same PORTABLE CPU build as
`cgs build --env PORTABLE=1`). It then checks that a `.deb`, `.rpm` and
`.AppImage` were all made, that the udev rule is inside the `.deb`, and that the
`.deb` has the ggml libraries and CPU plugins and its binary can find them.
Every other job compiles the whisper *stub*, so without this job a typo in a
feature, a broken `resources` path or a missing icon would pass CI and only
fail at release time.

**`.github/workflows/release.yml`** runs on a `v*` tag. It first fails fast if
the tag doesn't match the version in `tauri.conf.json`, `Cargo.toml` and
`desktop-ui/package.json`. Then it builds the Linux bundles and the Windows
installer, writes checksums, and opens a **draft** GitHub release with them
attached. The Windows job runs after the Linux one, so only one job creates the
draft, and Windows adds its files to it.

It stops at a draft on purpose. The updater's private key never goes into CI,
so `latest.json` doesn't exist yet. You finish by hand; see [In-app
updater](#in-app-updater) below. The updater can't see a release published
without `latest.json`.

> Status: both **Linux CPU** and **Windows CPU** builds are made. GPU support
> reaches users as the downloadable CUDA plugin (see [GPU_PLUGIN.md](GPU_PLUGIN.md)),
> built by its own workflow. Neither plugin archive is published yet.

## GPU plugin

Built and packaged separately from the application bundle:

```bash
cgs gpu-plugin        # -> release-artifacts/lune-voice-gpu-cuda13-linux-x86_64.tar.gz + .sha256
cgs stage-runtime     # copy ggml shared objects + backend plugins next to a built binary
```

Upload both files to the GitHub release, then record the digest and byte size in
`CUDA_LINUX_X86_64` (`crates/lune-persist/src/gpu_plugin.rs`). Until that digest
is filled in, the in-app install refuses instead of downloading something it
can't verify. The Windows plugin and the CI workflow are covered in
[GPU_PLUGIN.md](GPU_PLUGIN.md).

## In-app updater

> **On, but checks fail for now.** `UPDATES_ENABLED` in
> `desktop-ui/src-tauri/src/update.rs` is `true`. The endpoint in
> `tauri.conf.json` is a release file on a private repository, and the updater
> sends no credentials, so GitHub answers every check with a 404. A check you
> start ends in **Update check failed** in the update dialog, with the error
> text. The launch check stays quiet. Setting the constant to `false` hides the
> feature, and the app then says updates are off.
>
> The rest of this section describes how updates work and how to publish them.
> Once the release files can be downloaded without logging in, checks work with
> no code change. The GPU plugin pin waits on the same thing
> ([GPU_PLUGIN.md](GPU_PLUGIN.md)). Signature checking doesn't depend on any of
> this and is still tested (`cargo test -p lune-ui --lib updater`).

The tray's **Check for Updates…** fetches `latest.json` from the latest GitHub
release, verifies the signature against the public key pinned in
`tauri.conf.json`, and installs it. An AppImage is replaced in place; a
`.deb`/`.rpm` goes through `pkexec` (a graphical password prompt). Nothing is
downloaded until the user agrees, and the app restarts itself afterwards.

**Everything rests on the signing key.** It lives at
`~/.tauri/lune-voice-updater.key` (mode 600), never in the repo. Anyone who has
it can push a signed update to every user. If you lose it, no existing install
can ever update again, because the public key is built into shipped binaries.
Back it up somewhere safe, and keep it out of `git` and CI logs.

Builds don't need the key, on purpose. Turning on `createUpdaterArtifacts` would
make the bundler ask for the private key on **every** `cgs build`: a pinned
`pubkey` without its private key stops `tauri build` cold (*"A public key has
been found, but no private key"*), which would break the build for anyone else.
So nothing is signed while bundling. Afterwards, `cgs updater-manifest` packs
the AppImage into the `.tar.gz` the updater expects and signs all three formats
in one place.

For each release, bump the version in `tauri.conf.json`, `Cargo.toml` and
`desktop-ui/package.json`, then push the tag. `release.yml` builds the bundles
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
and download steps, but not the `rm -rf`. `cgs package` writes into the same
folder, and old release files left there are exactly what the manifest's
duplicate check rejects. `cgs package` also builds only Linux files, so the
resulting `latest.json` has no `windows-x86_64` key.

Upload `latest.json` to the release the endpoint points to, or users keep
seeing the previous version. Bump `version` in `tauri.conf.json` before
building: the updater compares it with the running app's version, so a release
built with an old version number is invisible.

The manifest has one key per package format
(`linux-x86_64-{appimage,deb,rpm}`), because the updater rejects a file that
doesn't match how the app was installed. One URL can't serve all three.
Windows has a single installer and so a single key, `windows-x86_64`.
The updater finds these keys by itself. It tries `{os}-{arch}-{installer}` for
the format this copy was installed from, then falls back to the plain
`{os}-{arch}` key, so the app doesn't need to name a target. The AppImage is
sent as a `.tar.gz` (the updater unpacks it and looks for the `.AppImage`
inside). `cgs updater-manifest` builds and signs that archive, and also signs the
deb and rpm, which the bundler wouldn't sign even with `createUpdaterArtifacts`
on.

`cgs updater-manifest` won't run when the tag doesn't match `version` in
`tauri.conf.json`, and it picks files **by that version**, not by name order.
Name order sorts the oldest version first, so it would pick 0.1.0 files for a
0.2.0 manifest. This check gives two errors:

- `no <kind> for <version> in <dir>, only: …`: the folder has that format, but
  for another version. Almost always a missing `rm -rf release-artifacts`.
- `more than one <kind> for <version>`: two builds of the same format and
  version, such as a `-rc.1` next to the final. Remove the one you aren't
  shipping.

Only the packed `.AppImage.tar.gz` and the portable archive may be absent.

### Testing the updater

Two levels of testing, both local. Neither needs a published release:

```bash
cargo test -p lune-ui --lib updater   # check + download + signature verification
cgs updater-test                      # the install path, for real
```

The unit tests serve a file signed with a throwaway key from a local HTTP
server and check three things: a correctly signed file is accepted, a file with
one flipped byte is refused, and a file signed by a *different valid key* is
refused. The last two matter most, because they prove the check is tied to our
key and isn't just checking the format.

`cgs updater-test` covers what unit tests can't: replacing the running app. It
builds the current version and a `99.0.0` AppImage, serves a signed manifest on
localhost, and prints the command to run the old one against it. Both bundles
are `--debug` on purpose. Release builds refuse plain-HTTP endpoints, so testing
over HTTP doesn't require loosening the shipped config, and the signature is
still checked against the pinned key. `LUNE_UPDATER_ENDPOINT` points a build at
the local manifest. It is compiled out of release builds, so a shipped binary
always uses the endpoint in `tauri.conf.json`.

### How a found update is announced

It depends on who asked.

A check the user started (the tray's *Check for Updates…*, or the button in
Settings → General) always answers in a dialog, including *"you are on the
latest version"* and every failure. A button that does nothing visible looks
broken.

A check started by **Check for updates on launch** (Settings → General, off by
default) shows a **desktop notification** instead, through
`tauri-plugin-notification`: libnotify on Linux, a toast on Windows. Neither
needs a permission prompt. Twenty seconds after login the user is busy in
another window, and a dialog stealing focus there would be more annoying than
the update is urgent. The notification only announces: *Lune Voice 1.4.0 is
available* / *You're running 1.3.2.* No desktop backend supports a click
action (the plugin only has notification actions on Android and iOS), and
writing out where to click would turn a quick glance into instructions. A
launch check that finds *nothing*, or fails, stays quiet, since having no
network a second after login is normal.

**There is no way to confirm a notification was shown.** Every desktop backend
hands it to a background task and returns `Ok` right away, so a session with no
notification daemon looks the same as one that showed it. The dialog fallback in
`notify_update_available` only catches the plugin *rejecting* the call, not a
notification nobody saw. A user with no notification daemon gets no launch
announcement, and finds the update the next time they check by hand. Fixing
that properly needs a lasting marker in the app (say, a tray item that reads
*Install 1.4.0…* until used), not a better notification.

`cargo test -p lune-ui --lib notification_tests` covers what can be checked
without a daemon: the text names the new and installed versions the right way
round, the plugin accepts the builder calls, and each `CheckMode` leads to the
right kind of announcement. To see a real notification, turn the setting on and
run the `cgs updater-test` build against the local endpoint.

## Third-party notices

`whisper.cpp` and `ggml` are MIT-licensed. They are statically linked into the
default (native) `cgs build`, and shipped as separate shared libraries in every
release bundle and anywhere else `GGML_BACKEND_DL` is on. The GPU plugin
archive also contains NVIDIA's redistributable CUDA runtime libraries. Update
`THIRD_PARTY_NOTICES.md` whenever the bundled upstream version changes.

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
with an out-of-date file fails instead of shipping wrong notices.
`build-windows` depends on it through `needs:`. cargo-about runs with `--fail`:
a crate whose licence it can't work out stops the run and needs a `clarify`
entry in `about.toml`.

**Nothing checks this outside a tag build.** `ci.yml` has no licence step and
`cgs package` does not call the script, so a dependency change can land on
`main`, and a local bundle can be built, with a stale `THIRD_PARTY_LICENSES.md`.
Run `cgs licenses --env CHECK=1` yourself after touching `Cargo.toml` or
`package.json`.

Lune Voice itself is GPL-3.0-or-later; `cargo deny check licenses` gates the
dependency tree against the allow-list in `deny.toml`.

## Rollback

Rolling back means reinstalling the previous package. There is no server and no
shared migration, so the only question is what the newer version changed on
disk before you rolled back.

### What a rollback has to survive

| State | Rollback-safe? | Why |
|---|---|---|
| `settings.toml` | Mostly | Unknown keys are ignored on load, so a newer file loads on an older build. Fields the older build does not know are **dropped on its next save**. |
| `lune.db` | Only within the same schema version | Diesel records applied migrations. An older binary will not downgrade the schema on its own. |
| Models | Yes | Plain files, keyed by directory name. |
| GPU plugin | Yes | Independent of the app version; delete the directory to revert to CPU. |
| Keyring secrets | Yes | Slot names are stable and versionless. |

### Procedure

1. **Back up first.** This is what lets you undo the rollback:

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
   mean the database is ahead of the binary. Migrations that only add things
   (every one shipped so far) are fine, since the old code ignores the new
   columns. If the old build won't start, revert the extra migrations instead
   of deleting the database:

   ```sh
   diesel migration revert --migration-dir crates/lune-persist/migrations
   ```

   Each migration's `down.sql` is exercised by
   `migrations_revert_and_reapply_cleanly` in `crates/lune-persist/src/db/mod.rs`,
   so reverting is tested.

4. **Check** that dictation works end to end before calling it done:

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

An older build quietly drops settings fields it doesn't know, so a release that
**removes or renames** a `settings.toml` field can't be safely rolled back.
Move such a field in a release that still reads both names, and drop the old
name one release later.
