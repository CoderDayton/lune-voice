#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

# Build the desktop app for distribution. Arguments go to the final
# `tauri build`, e.g. `--bundles deb,appimage,rpm` or `--no-bundle`.
#
# ggml is built as shared libraries with its compute backends as plugins
# (GGML_BACKEND_DL), and the bundle ships two CPU backends, which ggml chooses
# between at startup:
#   haswell  AVX2 + FMA + F16C + BMI2 — Intel since 2013, AMD since 2015
#   sse42    no AVX — every CPU the x86-64-v2 Rust baseline runs on
# ggml has no switch for a chosen subset, so GGML_CPU_ALL_VARIANTS builds every
# x86 variant and only these two are staged. One build therefore runs on any
# x86-64-v2 CPU and can load the downloadable CUDA plugin (docs/GPU_PLUGIN.md).
#
# Two passes, because the bundle carries files the first pass produces:
#   1. `tauri build --no-bundle` compiles whisper.cpp and the app.
#   2. The ggml libraries and CPU plugins are staged into
#      desktop-ui/src-tauri/runtime/, and `tauri build` runs again with
#      tauri.runtime.conf.json, which bundles that directory as resources.
#      Only the app crate recompiles.
#
# Installed layout:
#   Linux    <prefix>/lib/LuneVoice/{libwhisper,libparakeet,libggml,libggml-base}.so.N
#            <prefix>/lib/LuneVoice/backends/libggml-cpu-{sse42,haswell}.so
#   Windows  <install dir>/{whisper,parakeet,ggml,ggml-base}.dll
#            <install dir>/backends/ggml-cpu-{sse42,haswell}.dll
#
# Honours CARGO_TARGET_DIR. Runs under Git Bash on Windows.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UI="$REPO_ROOT/desktop-ui"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
RUNTIME="$UI/src-tauri/runtime"
FEATURES="whisper-cpp"
CPU_VARIANTS=(sse42 haswell)

export GGML_BACKEND_DL=ON GGML_NATIVE=OFF GGML_CPU_ALL_VARIANTS=ON
# ggml picks its CPU code at runtime; the Rust half is compiled once, for the
# oldest CPU the release supports.
case "$(uname -m)" in
    x86_64 | amd64) export RUSTFLAGS="-C target-cpu=x86-64-v2" ;;
    *) export RUSTFLAGS="" ;;
esac

echo "==> Pass 1: compile ($FEATURES, RUSTFLAGS=\"$RUSTFLAGS\")"
(cd "$UI" && bun tauri build --no-bundle --features "$FEATURES")

# The newest whisper-rs-sys build output that holds shared libraries. Cargo can
# keep several; an older one may be a static build.
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) windows=1 marker="bin/ggml-base.dll" ;;
    *) windows=0 marker="lib/libggml-base.so" ;;
esac
out=""
for d in "$TARGET_DIR"/release/build/whisper-rs-sys-*/out; do
    [[ -e "$d/$marker" ]] || continue
    if [[ -z "$out" || "$d/$marker" -nt "$out/$marker" ]]; then out="$d"; fi
done
if [[ -z "$out" ]]; then
    echo "ERROR: no shared whisper-rs-sys build under $TARGET_DIR/release/build" >&2
    exit 1
fi

echo "==> Staging ggml runtime from $out"
rm -rf "$RUNTIME"
mkdir -p "$RUNTIME/backends"
shopt -s nullglob
backends=()
if [[ "$windows" == 1 ]]; then
    for lib in whisper parakeet ggml ggml-base; do
        cp "$out/bin/$lib.dll" "$RUNTIME/"
    done
    for v in "${CPU_VARIANTS[@]}"; do backends+=("$out/bin/ggml-cpu-$v.dll"); done
else
    # The executable needs each library under its SONAME (`libggml.so.0`); the
    # other names are symlinks, which bundles do not keep.
    for lib in libwhisper libparakeet libggml libggml-base; do
        sonames=()
        for f in "$out/lib/$lib".so.*; do
            [[ "$(basename "$f")" =~ ^$lib\.so\.[0-9]+$ ]] && sonames+=("$f")
        done
        if [[ ${#sonames[@]} -ne 1 ]]; then
            echo "ERROR: expected one $lib.so.N in $out/lib, found ${#sonames[@]}" >&2
            exit 1
        fi
        cp -L "${sonames[0]}" "$RUNTIME/"
    done
    for v in "${CPU_VARIANTS[@]}"; do backends+=("$out/bin/libggml-cpu-$v.so"); done
fi
shopt -u nullglob
for f in "${backends[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: CPU backend $(basename "$f") missing from $out/bin" >&2
        exit 1
    fi
done
cp "${backends[@]}" "$RUNTIME/backends/"
echo "    staged: $(cd "$RUNTIME" && echo *) + backends: ${CPU_VARIANTS[*]}"

echo "==> Pass 2: bundle"
(cd "$UI" && bun tauri build --features "$FEATURES" --config src-tauri/tauri.runtime.conf.json "$@")
