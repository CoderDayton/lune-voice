#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

# Build the shippable Linux artifacts in one container: .deb, .AppImage, .rpm
# and a portable .tar.gz.
#
# Usage:
#   bash scripts/release/package-all.sh            # bundles + tarball
#   bash scripts/release/package-all.sh bundles    # deb + AppImage + rpm only
#   bash scripts/release/package-all.sh tarball    # repack an existing .deb
#
# One image, not one per package format: tauri writes RPMs with the pure-Rust
# `rpm` crate, so it needs no rpmbuild and no Fedora. Ubuntu 22.04 is the right
# single host because it has the OLDEST glibc of the distros we target — a
# binary built on Fedora would not run on Ubuntu 22.04, but this one runs on
# both. The rpm's Requires come from tauri.conf.json, not from the build host,
# so they stay correct regardless.
#
# Honors $CONTAINER_ENGINE (docker|podman). Auto-detects if unset.
# Caches Rust toolchains, cargo registry, target/, and Bun across runs in
# $XDG_CACHE_HOME/lune-voice-build (saves ~10 min on warm rebuilds).
#
# Caveat: depends on `libwebkit2gtk-4.1-dev` being available on Ubuntu 22.04
# (jammy-updates pocket, default in the official ubuntu:22.04 image). If the
# apt step fails with "Unable to locate package", add the Tauri webkit-4.1
# PPA as a preflight step before the install line in build_bundles().

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACTS_DIR="$REPO_ROOT/release-artifacts"
mkdir -p "$ARTIFACTS_DIR"

CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/lune-voice-build"
CARGO_HOME_CACHE="$CACHE_ROOT/cargo"
RUSTUP_HOME_CACHE="$CACHE_ROOT/rustup"
BUN_HOME_CACHE="$CACHE_ROOT/bun"
TARGET_UBUNTU="$CACHE_ROOT/target-ubuntu22"
mkdir -p "$CARGO_HOME_CACHE" "$RUSTUP_HOME_CACHE" "$BUN_HOME_CACHE" "$TARGET_UBUNTU"

ENGINE="${CONTAINER_ENGINE:-}"
if [[ -z "$ENGINE" ]]; then
    if command -v docker &>/dev/null; then ENGINE=docker
    elif command -v podman &>/dev/null; then ENGINE=podman
    else
        echo "ERROR: install docker or podman, or set CONTAINER_ENGINE" >&2
        exit 1
    fi
fi

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

WANT_BUNDLES=1
WANT_TARBALL=1
case "${1:-all}" in
    all)                                  ;;
    bundles|deb|appimage|rpm)  WANT_TARBALL=0 ;;
    # Repacks whatever .deb is already in release-artifacts/ — no container run.
    tarball|tar.gz)            WANT_BUNDLES=0 ;;
    *) echo "Usage: $0 [all|bundles|tarball]" >&2; exit 1 ;;
esac

build_bundles() {
    bold "==> DEB + AppImage + RPM (Ubuntu 22.04 container)"
    # A nameref-free array keeps each path quoted as one argv element, so paths
    # with spaces survive.
    local flags=(
        --rm
        -i
        -v "$REPO_ROOT:/work"
        -v "$CARGO_HOME_CACHE:/root/.cargo"
        -v "$RUSTUP_HOME_CACHE:/root/.rustup"
        -v "$BUN_HOME_CACHE:/root/.bun"
        -v "$TARGET_UBUNTU:/cache/target"
        -e CARGO_TARGET_DIR=/cache/target
        -e CARGO_INCREMENTAL=0
        -e RUSTFLAGS=
        -e APPIMAGE_EXTRACT_AND_RUN=1
        -e "HOST_UID=$HOST_UID"
        -e "HOST_GID=$HOST_GID"
        -w /work
    )

    # The inner script goes to the container's stdin, so $PATH / $HOST_UID are
    # resolved by the container's shell, not the host's.
    "$ENGINE" run "${flags[@]}" ubuntu:22.04 bash -euo pipefail <<'INNER'
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential cmake libclang-dev pkg-config curl ca-certificates file desktop-file-utils unzip \
    libasound2-dev libdbus-1-dev libglib2.0-dev libssl-dev \
    libxkbcommon-dev libgtk-3-dev libgtk-layer-shell-dev \
    libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev \
    libsoup-3.0-dev librsvg2-dev libayatana-appindicator3-dev

export PATH="/root/.bun/bin:/root/.cargo/bin:$PATH"

if [[ ! -x /root/.cargo/bin/cargo ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
fi

if [[ ! -x /root/.bun/bin/bun ]]; then
    curl -fsSL https://bun.sh/install | bash
fi

# Restore host ownership of cached/output dirs whether the build succeeds or
# fails — without this trap, an aborted run leaves root-owned files in
# /work/desktop-ui/{dist,node_modules}, blocking subsequent host builds.
restore_ownership() {
    chown -R "$HOST_UID:$HOST_GID" \
        /cache/target \
        /root/.cargo \
        /root/.rustup \
        /root/.bun \
        /work/desktop-ui/node_modules \
        /work/desktop-ui/dist 2>/dev/null || true
}
trap restore_ownership EXIT

cd /work/desktop-ui
bun install --frozen-lockfile
# --features whisper-cpp compiles the local whisper.cpp backend into the shipped
# bundle. The GGML_* flags force a portable AVX2 CPU baseline instead of ggml's
# default -march=native, which would SIGILL on user CPUs older than the builder.
# NOTE: these only apply on a FRESH whisper-rs-sys build — its build.rs has no
# rerun-if-env-changed, so cmake's cache wins on a warm target. Release container
# caches start clean for whisper (correct on first build); if you change these
# flags later, wipe the cached target dirs first.
GGML_NATIVE=OFF GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON \
    bun tauri build --bundles deb,appimage,rpm --features whisper-cpp
INNER

    local count=0 f dest
    shopt -s nullglob
    for f in "$TARGET_UBUNTU/release/bundle/deb/"*.deb \
             "$TARGET_UBUNTU/release/bundle/appimage/"*.AppImage \
             "$TARGET_UBUNTU/release/bundle/rpm/"*.rpm; do
        # Tauri names bundles after `productName`, which is "Lune Voice" — a
        # space in a download URL becomes %20 and breaks naive installers and
        # shell one-liners. The name inside the package is already `lune-voice`,
        # so this only fixes the file name.
        dest="$(basename "$f")"
        dest="${dest//Lune Voice/lune-voice}"
        cp "$f" "$ARTIFACTS_DIR/$dest"
        count=$((count + 1))
    done
    shopt -u nullglob

    if [[ $count -eq 0 ]]; then
        red "WARN: no bundles produced in $TARGET_UBUNTU/release/bundle/"
    else
        green "Bundles: $count artifact(s)."
    fi
}

# The portable archive is the .deb payload, not a second compile: same binary,
# same old-glibc baseline, minus dpkg. Tauri resolves its resource directory
# relative to the executable's prefix (`<prefix>/bin/lune-voice` ->
# `<prefix>/lib/Lune Voice`), so keeping the deb's `usr/` tree intact is what
# lets an extracted copy find its resources with no install step.
build_tarball() {
    shopt -s nullglob
    local debs=("$ARTIFACTS_DIR"/*.deb)
    shopt -u nullglob
    if [[ ${#debs[@]} -eq 0 ]]; then
        red "WARN: no .deb in $ARTIFACTS_DIR to repack — skipping tar.gz"
        return
    fi
    if ! command -v dpkg-deb &>/dev/null; then
        red "WARN: dpkg-deb not installed — skipping tar.gz"
        return
    fi

    # Newest .deb wins, so a stale one from an earlier version is not repacked.
    # Capture the listing, then take its first line: `ls -t | head -1` under
    # `pipefail` is the same SIGPIPE race the CI udev check hit — head exits on
    # line one, ls dies of SIGPIPE, and the substitution fails a step that was
    # working.
    local deb listing
    listing="$(ls -t "${debs[@]}")"
    deb="${listing%%$'\n'*}"
    bold "==> Portable tar.gz (from $(basename "$deb"))"

    local version arch name stage
    version="$(dpkg-deb -f "$deb" Version)"
    arch="$(dpkg-deb -f "$deb" Architecture)"
    name="lune-voice-${version}-linux-${arch}"

    stage="$(mktemp -d)"
    dpkg-deb -x "$deb" "$stage/$name"

    if [[ ! -x "$stage/$name/usr/bin/lune-voice" ]]; then
        red "WARN: no usr/bin/lune-voice inside the .deb — skipping tar.gz"
        rm -rf "$stage"
        return
    fi

    # The bundler stamps the binary with the package it went into. Left as the
    # .deb stamp, the updater installs the next version with `dpkg -i` through
    # pkexec, asking root of someone who chose a no-root archive. Restore the
    # "no bundle" stamp (same length) so the app offers a manual download.
    local bin="$stage/$name/usr/bin/lune-voice" stamps
    stamps="$(grep -a -o '__TAURI_BUNDLE_TYPE_VAR_DEB' "$bin" | wc -l)"
    if [[ "$stamps" -ne 1 ]]; then
        red "WARN: expected one .deb bundle stamp in the binary, found $stamps — skipping tar.gz"
        rm -rf "$stage"
        return
    fi
    perl -0777 -pi -e 's/__TAURI_BUNDLE_TYPE_VAR_DEB/__TAURI_BUNDLE_TYPE_VAR_UNK/' "$bin"

    # Top-level launcher: resolves its own directory, so the archive runs from
    # wherever it was extracted.
    cat > "$stage/$name/lune-voice" <<'LAUNCHER'
#!/usr/bin/env sh
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/usr/bin/lune-voice" "$@"
LAUNCHER
    chmod +x "$stage/$name/lune-voice"

    # The udev rule is the one thing extraction cannot do for the user: without
    # it /dev/uinput is root-only and keystroke delivery silently fails.
    cat > "$stage/$name/README.txt" <<README
Lune Voice $version — portable Linux build ($arch)

Run it:

    ./lune-voice

Text delivery needs write access to /dev/uinput. The .deb and .rpm set this up;
for this archive, install the rule once and log back in:

    sudo cp usr/lib/udev/rules.d/99-lune-voice-uinput.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger

Speech models are downloaded by the app on first run into
~/.local/share/lune-voice/models/ and are not part of this archive.

This archive does not update itself. When a new version is out, the app says
so; download the new archive and extract it over this one.

Licences: usr/lib/Lune Voice/THIRD_PARTY_NOTICES.md
README

    tar -czf "$ARTIFACTS_DIR/$name.tar.gz" -C "$stage" "$name"
    ( cd "$ARTIFACTS_DIR" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256" )
    rm -rf "$stage"

    green "tar.gz: $name.tar.gz ($(du -h "$ARTIFACTS_DIR/$name.tar.gz" | cut -f1))"
}

[[ "$WANT_BUNDLES" == 1 ]] && build_bundles
[[ "$WANT_TARBALL" == 1 ]] && build_tarball

bold "==> Artifacts:"
ls -lh "$ARTIFACTS_DIR" || true
