#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build the downloadable CUDA backend plugin and package it for GitHub Releases.
#
# The archive is deliberately self-contained apart from the NVIDIA *driver*:
# libcudart/libcublas/libcublasLt are redistributable under the CUDA EULA,
# libcuda.so.1 ships with the driver and is not ours to distribute. The plugin
# is linked with RUNPATH=$ORIGIN so it resolves those runtime libraries from its
# own directory — LD_LIBRARY_PATH cannot be used here, because glibc caches the
# search path at process start and the app sets nothing before dlopen.
set -euo pipefail

OUT="${OUT:-release-artifacts}"
BUILD_DIR="${BUILD_DIR:-target/gpu-plugin}"
SRC="vendor/whisper-rs-sys/whisper.cpp"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/targets/x86_64-linux/lib}"
# Native SASS for the four consumer generations CUDA 13 supports (Turing,
# Ampere, Ada, Blackwell), plus compute_75 PTX as a floor. PTX JITs *forward*
# only, so the oldest virtual arch covers every other supported GPU — A100,
# H100, datacentre Blackwell — at the cost of a one-off JIT on first run.
# Each extra -real arch adds its own SASS to the download; override to trade
# coverage for size.
CUDA_ARCHS="${CUDA_ARCHS:-75-real;86-real;89-real;120-real;75-virtual}"

command -v cmake    >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v nvcc     >/dev/null || { echo "the CUDA toolkit (nvcc) is required" >&2; exit 1; }
command -v cuobjdump >/dev/null || { echo "cuobjdump (CUDA toolkit) is required" >&2; exit 1; }
[ -d "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }

# CMake caches CMAKE_CUDA_ARCHITECTURES, so reusing a build dir configured for a
# narrower list silently keeps producing that narrower plugin. Reconfigure from
# scratch whenever the list changes.
CACHED_ARCHS=$(grep -oP '^CMAKE_CUDA_ARCHITECTURES:[^=]*=\K.*' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)
if [ -n "$CACHED_ARCHS" ] && [ "$CACHED_ARCHS" != "$CUDA_ARCHS" ]; then
  echo "CUDA architectures changed ($CACHED_ARCHS -> $CUDA_ARCHS); reconfiguring"
  rm -rf "$BUILD_DIR"
fi

CUDA_VER=$(basename "$(readlink -f "$CUDA_LIB"/libcudart.so.* | head -1)" | sed 's/libcudart\.so\.//' | cut -d. -f1)
[ -n "$CUDA_VER" ] || { echo "could not determine the CUDA runtime soname version" >&2; exit 1; }
echo "Building CUDA $CUDA_VER backend plugin for archs $CUDA_ARCHS"

cmake -S "$SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' >/dev/null
cmake --build "$BUILD_DIR" --target ggml-cuda -j "${JOBS:-$(nproc)}"

# `-not -path '*/stage/*'`: the previous run's copy lives under $STAGE, which is
# wiped below — matching it here would delete the file and then fail to copy it.
PLUGIN=$(find "$BUILD_DIR" -not -path '*/stage/*' -name 'libggml-cuda.so' -type f | head -1)
[ -n "$PLUGIN" ] || { echo "libggml-cuda.so was not produced" >&2; exit 1; }

STAGE="$BUILD_DIR/stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -a "$PLUGIN" "$STAGE/"
for lib in libcudart libcublas libcublasLt; do
  src=$(ls "$CUDA_LIB/$lib.so.$CUDA_VER" 2>/dev/null | head -1)
  [ -n "$src" ] || { echo "missing $lib.so.$CUDA_VER in $CUDA_LIB" >&2; exit 1; }
  cp -aL "$src" "$STAGE/"
done

# The app loads the plugin only when this stamp equals its own `ggml_version()`:
# a plugin built against another ggml registers its device and then crashes.
GGML_VERSION=$(sed -nE 's/^set\(GGML_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)\)$/\2/p' "$SRC/ggml/CMakeLists.txt" | paste -sd.)
[[ "$GGML_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "could not read the ggml version from $SRC/ggml/CMakeLists.txt" >&2; exit 1; }
printf '%s\n' "$GGML_VERSION" > "$STAGE/ggml-version"

# Debug info in the plugin is dead weight in a download; NVIDIA's libraries
# arrive stripped already. nvprune cannot help with the rest: it only operates
# on relocatable device code, and the shipped cuBLAS objects are final.
strip --strip-unneeded "$STAGE/libggml-cuda.so" 2>/dev/null || true

# The whole scheme rests on this: without $ORIGIN the plugin cannot find the
# CUDA runtime sitting beside it, and dlopen fails at load.
if ! readelf -d "$STAGE/libggml-cuda.so" | grep -qE 'R(UN)?PATH.*\$ORIGIN'; then
  echo "libggml-cuda.so has no \$ORIGIN runpath — it would not resolve its CUDA runtime" >&2
  exit 1
fi

# Assert against the built binary, not the configure step. A GPU whose arch is
# absent from both the SASS and the PTX cannot run the plugin at all, and that
# is invisible until a user with that card tries it.
SASS=$(cuobjdump -lelf "$STAGE/libggml-cuda.so" 2>/dev/null | { grep -oP 'sm_\K[0-9]+[a-z]*' || true; } | sort -u)
PTX=$(cuobjdump  -lptx "$STAGE/libggml-cuda.so" 2>/dev/null | { grep -oP 'sm_\K[0-9]+[a-z]*' || true; } | sort -u)
MISSING=""
IFS=';' read -ra WANT <<< "$CUDA_ARCHS"
for arch in "${WANT[@]}"; do
  # ggml rewrites 12X into the architecture-specific 12Xa (Blackwell's FP4
  # tensor-core instructions are not forward-compatible), so a request for 120
  # is satisfied by sm_120a. Match the number and allow that suffix.
  case "$arch" in
    *-virtual) want="${arch%-virtual}"; have="$PTX"  ;;
    *)         want="${arch%-real}";    have="$SASS" ;;
  esac
  grep -qE "^${want%[a-z]}[a-z]?$" <<< "$have" || MISSING="$MISSING $arch"
done
[ -z "$MISSING" ] || { echo "plugin is missing requested architectures:$MISSING" >&2; exit 1; }
echo "verified SASS: $(echo $SASS | tr '\n' ' ') | PTX: $(echo $PTX | tr '\n' ' ')"

mkdir -p "$OUT"
NAME="lune-voice-gpu-cuda${CUDA_VER}-linux-x86_64"
tar -czf "$OUT/$NAME.tar.gz" -C "$STAGE" .
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )

echo
echo "  $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
cat "$OUT/$NAME.tar.gz.sha256"
echo
echo "Upload both files to the GitHub release, then record the sha256 and the"
echo "byte size in CUDA_LINUX_X86_64 (crates/lune-persist/src/gpu_plugin.rs)"
echo "so the download is verified."
