#!/usr/bin/env bash
set -euo pipefail

# Build .deb / .AppImage / .rpm bundles via containerized cross-distro builds.
# Usage:
#   bash scripts/package-all.sh                 # all targets
#   bash scripts/package-all.sh deb-appimage    # Ubuntu 22.04 only
#   bash scripts/package-all.sh rpm             # Fedora 40 only
#
# Honors $CONTAINER_ENGINE (docker|podman). Auto-detects if unset.
# Caches Rust toolchains, cargo registry, target/, and Bun across runs in
# $XDG_CACHE_HOME/lune-voice-build (saves ~10 min on warm rebuilds).
#
# Caveat: depends on `libwebkit2gtk-4.1-dev` being available on Ubuntu 22.04
# (jammy-updates pocket, default in the official ubuntu:22.04 image). If the
# apt step fails with "Unable to locate package", add the Tauri webkit-4.1
# PPA as a preflight step before the install line in build_deb_appimage().

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$REPO_ROOT/release-artifacts"
mkdir -p "$ARTIFACTS_DIR"

CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/lune-voice-build"
CARGO_HOME_CACHE="$CACHE_ROOT/cargo"
RUSTUP_HOME_CACHE="$CACHE_ROOT/rustup"
BUN_HOME_CACHE="$CACHE_ROOT/bun"
TARGET_UBUNTU="$CACHE_ROOT/target-ubuntu22"
TARGET_FEDORA="$CACHE_ROOT/target-fedora40"
mkdir -p "$CARGO_HOME_CACHE" "$RUSTUP_HOME_CACHE" "$BUN_HOME_CACHE" \
         "$TARGET_UBUNTU" "$TARGET_FEDORA"

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

WANT_DEB_APPIMAGE=1
WANT_RPM=1
case "${1:-all}" in
    all)                          ;;
    deb-appimage|deb|appimage)    WANT_RPM=0 ;;
    rpm)                          WANT_DEB_APPIMAGE=0 ;;
    *) echo "Usage: $0 [all|deb-appimage|rpm]" >&2; exit 1 ;;
esac

# Populate the named bash array with common docker run flags. Using a nameref
# keeps each path quoted as a single argv element — robust to spaces in paths.
populate_run_flags() {
    local -n out=$1
    local target_cache=$2
    # shellcheck disable=SC2034  # `out` is a nameref to caller's array
    out=(
        --rm
        -i
        -v "$REPO_ROOT:/work"
        -v "$CARGO_HOME_CACHE:/root/.cargo"
        -v "$RUSTUP_HOME_CACHE:/root/.rustup"
        -v "$BUN_HOME_CACHE:/root/.bun"
        -v "$target_cache:/cache/target"
        -e CARGO_TARGET_DIR=/cache/target
        -e CARGO_INCREMENTAL=0
        -e RUSTFLAGS=
        -e APPIMAGE_EXTRACT_AND_RUN=1
        -e "HOST_UID=$HOST_UID"
        -e "HOST_GID=$HOST_GID"
        -w /work
    )
}

# Inner build script — emitted to the container's stdin, which means $PATH /
# $BUNDLES / $HOST_UID etc. are resolved by the *container's* shell, not the
# host's. `$BUNDLES` is injected via `docker run -e BUNDLES=...` by the caller.
build_inner() {
    cat <<'INNER'
export PATH="/root/.bun/bin:/root/.cargo/bin:$PATH"

if [[ ! -x /root/.cargo/bin/cargo ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
fi

if [[ ! -x /root/.bun/bin/bun ]]; then
    curl -fsSL https://bun.sh/install | bash
fi

cd /work/desktop-ui
bun install --frozen-lockfile
bun tauri build --bundles "$BUNDLES"

# Restore host ownership of cached/output dirs before the container exits.
chown -R "$HOST_UID:$HOST_GID" \
    /cache/target \
    /root/.cargo \
    /root/.rustup \
    /root/.bun \
    /work/desktop-ui/node_modules \
    /work/desktop-ui/dist 2>/dev/null || true
INNER
}

# Copy artifacts into release-artifacts/ and emit count via stdout.
collect_artifacts() {
    local count=0
    local f
    shopt -s nullglob
    for f in "$@"; do
        cp "$f" "$ARTIFACTS_DIR/"
        count=$((count + 1))
    done
    shopt -u nullglob
    printf '%d\n' "$count"
}

build_deb_appimage() {
    bold "==> DEB + AppImage (Ubuntu 22.04 container)"
    local flags
    populate_run_flags flags "$TARGET_UBUNTU"
    {
        cat <<'PRE'
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential pkg-config curl ca-certificates file desktop-file-utils \
    libasound2-dev libdbus-1-dev libglib2.0-dev libssl-dev \
    libxkbcommon-dev libgtk-3-dev \
    libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev \
    libsoup-3.0-dev librsvg2-dev libayatana-appindicator3-dev
PRE
        build_inner
    } | "$ENGINE" run "${flags[@]}" \
        -e BUNDLES=deb,appimage \
        ubuntu:22.04 \
        bash -euo pipefail

    local n
    shopt -s nullglob
    local files=(
        "$TARGET_UBUNTU/release/bundle/deb/"*.deb
        "$TARGET_UBUNTU/release/bundle/appimage/"*.AppImage
    )
    shopt -u nullglob
    n=$(collect_artifacts "${files[@]}")
    if [[ $n -eq 0 ]]; then
        red "WARN: no .deb/.AppImage produced in $TARGET_UBUNTU/release/bundle/"
    else
        green "DEB + AppImage: $n artifact(s)."
    fi
}

build_rpm() {
    bold "==> RPM (Fedora 40 container)"
    local flags
    populate_run_flags flags "$TARGET_FEDORA"
    {
        cat <<'PRE'
dnf install -y \
    gcc gcc-c++ make pkgconf-pkg-config curl file \
    openssl-devel alsa-lib-devel dbus-devel glib2-devel \
    libxkbcommon-devel gtk3-devel \
    webkit2gtk4.1-devel javascriptcoregtk4.1-devel \
    libsoup3-devel librsvg2-devel libappindicator-gtk3-devel \
    rpm-build
PRE
        build_inner
    } | "$ENGINE" run "${flags[@]}" \
        -e BUNDLES=rpm \
        fedora:40 \
        bash -euo pipefail

    local n
    shopt -s nullglob
    local files=("$TARGET_FEDORA/release/bundle/rpm/"*.rpm)
    shopt -u nullglob
    n=$(collect_artifacts "${files[@]}")
    if [[ $n -eq 0 ]]; then
        red "WARN: no .rpm produced in $TARGET_FEDORA/release/bundle/rpm/"
    else
        green "RPM: $n artifact(s)."
    fi
}

[[ "$WANT_DEB_APPIMAGE" == 1 ]] && build_deb_appimage
[[ "$WANT_RPM" == 1 ]]          && build_rpm

bold "==> Artifacts:"
ls -lh "$ARTIFACTS_DIR" || true
