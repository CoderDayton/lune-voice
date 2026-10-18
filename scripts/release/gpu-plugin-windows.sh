#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

OUT="${OUT:-release-artifacts}"
BUILD_DIR="${BUILD_DIR:-target/gpu-plugin-windows}"
SRC="vendor/whisper-rs-sys/whisper.cpp"
CUDA_ROOT="${CUDA_PATH:-}"
CUDA_ARCHS="${CUDA_ARCHS:-75-real;86-real;89-real;120-real;75-virtual}"

command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v nvcc  >/dev/null || { echo "the CUDA toolkit (nvcc) is required" >&2; exit 1; }
[ -d "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
[ -n "$CUDA_ROOT" ] || { echo "CUDA_PATH is not set — run from a shell where the CUDA toolkit is configured" >&2; exit 1; }
command -v cygpath >/dev/null && CUDA_ROOT=$(cygpath -u "$CUDA_ROOT")
CUDA_BIN=""
for dir in "$CUDA_ROOT/bin/x64" "$CUDA_ROOT/bin"; do
  if ls "$dir"/cudart64_*.dll >/dev/null 2>&1; then CUDA_BIN="$dir"; break; fi
done
[ -n "$CUDA_BIN" ] || { echo "no cudart64_*.dll under $CUDA_ROOT/bin/x64 or $CUDA_ROOT/bin" >&2; exit 1; }

CACHED_ARCHS=$(grep -oP '^CMAKE_CUDA_ARCHITECTURES:[^=]*=\K.*' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)
if [ -n "$CACHED_ARCHS" ] && [ "$CACHED_ARCHS" != "$CUDA_ARCHS" ]; then
  echo "CUDA architectures changed ($CACHED_ARCHS -> $CUDA_ARCHS); reconfiguring"
  rm -rf "$BUILD_DIR"
fi

CUDART=$(ls "$CUDA_BIN"/cudart64_*.dll 2>/dev/null | head -1)
[ -n "$CUDART" ] || { echo "no cudart64_*.dll in $CUDA_BIN" >&2; exit 1; }
CUDA_VER=$(basename "$CUDART" | sed -E 's/cudart64_([0-9]+)\.dll/\1/')
[ -n "$CUDA_VER" ] || { echo "could not determine the CUDA runtime version from $CUDART" >&2; exit 1; }
echo "Building CUDA $CUDA_VER backend plugin for archs $CUDA_ARCHS"

cmake -S "$SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF >/dev/null
cmake --build "$BUILD_DIR" --config Release --target ggml-cuda -j "${JOBS:-$(nproc)}"

PLUGIN=$(find "$BUILD_DIR" -not -path '*/stage/*' -name 'ggml-cuda.dll' -type f | head -1)
[ -n "$PLUGIN" ] || { echo "ggml-cuda.dll was not produced" >&2; exit 1; }

STAGE="$BUILD_DIR/stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -a "$PLUGIN" "$STAGE/"
for lib in cudart cublas cublasLt; do
  src=$(ls "$CUDA_BIN/${lib}64_$CUDA_VER.dll" 2>/dev/null | head -1)
  [ -n "$src" ] || { echo "missing ${lib}64_$CUDA_VER.dll in $CUDA_BIN" >&2; exit 1; }
  cp -a "$src" "$STAGE/"
done

GGML_VERSION=$(sed -nE 's/^set\(GGML_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)\)$/\2/p' "$SRC/ggml/CMakeLists.txt" | paste -sd.)
[[ "$GGML_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "could not read the ggml version from $SRC/ggml/CMakeLists.txt" >&2; exit 1; }
printf '%s\n' "$GGML_VERSION" > "$STAGE/ggml-version"

for f in "$STAGE"/*; do
  base=$(basename "$f")
  case "$base" in
    ggml-cuda.dll|cudart64_*.dll|cublas64_*.dll|cublasLt64_*.dll|ggml-version) ;;
    *) echo "unexpected file in the archive: $base" >&2; exit 1 ;;
  esac
done

if command -v cuobjdump >/dev/null; then
  SASS=$(cuobjdump -lelf "$STAGE/ggml-cuda.dll" 2>/dev/null | { grep -oP 'sm_\K[0-9]+[a-z]*' || true; } | sort -u)
  PTX=$(cuobjdump  -lptx "$STAGE/ggml-cuda.dll" 2>/dev/null | { grep -oP 'sm_\K[0-9]+[a-z]*' || true; } | sort -u)
  MISSING=""
  IFS=';' read -ra WANT <<< "$CUDA_ARCHS"
  for arch in "${WANT[@]}"; do
    case "$arch" in
      *-virtual) want="${arch%-virtual}"; have="$PTX"  ;;
      *)         want="${arch%-real}";    have="$SASS" ;;
    esac
    grep -qE "^${want%[a-z]}[a-z]?$" <<< "$have" || MISSING="$MISSING $arch"
  done
  [ -z "$MISSING" ] || { echo "plugin is missing requested architectures:$MISSING" >&2; exit 1; }
  echo "verified SASS: $(echo $SASS | tr '\n' ' ') | PTX: $(echo $PTX | tr '\n' ' ')"
else
  echo "WARNING: cuobjdump not found — architecture coverage was NOT verified" >&2
fi

mkdir -p "$OUT"
NAME="lune-voice-gpu-cuda${CUDA_VER}-windows-x86_64"
tar -czf "$OUT/$NAME.tar.gz" -C "$STAGE" .
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )

echo
echo "  $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
cat "$OUT/$NAME.tar.gz.sha256"
echo
echo "Upload both files to the GitHub release, then record the sha256 and the"
echo "byte size in CUDA_WINDOWS_X86_64 (crates/lune-persist/src/gpu_plugin.rs)"
echo "and update the_cuda_windows_release_is_still_unpublished in that file."
