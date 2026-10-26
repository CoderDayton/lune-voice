#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

OUT="${OUT:-release-artifacts}"
BUILD_DIR="${BUILD_DIR:-target/gpu-plugin-vulkan}"
SRC="vendor/whisper-rs-sys/whisper.cpp"

case "$(uname -s)" in
  Linux) OS=linux; PLUGIN_FILE=libggml-vulkan.so ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows; PLUGIN_FILE=ggml-vulkan.dll ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
[ -d "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
[ -n "${VULKAN_SDK:-}" ] || { echo "VULKAN_SDK is not set — source the Vulkan SDK's setup-env.sh first" >&2; exit 1; }
SDK="$VULKAN_SDK"
command -v cygpath >/dev/null && SDK=$(cygpath -m "$SDK")

cmake -S "$SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_VULKAN=ON \
  -DCMAKE_PREFIX_PATH="$SDK;$SDK/lib/VulkanLoader" \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' >/dev/null
git checkout -- "$SRC/bindings/javascript/package.json" 2>/dev/null || true
cmake --build "$BUILD_DIR" --config Release --target ggml-vulkan -j "${JOBS:-$(nproc)}"

PLUGIN=$(find "$BUILD_DIR" -not -path '*/stage/*' -name "$PLUGIN_FILE" -type f | head -1)
[ -n "$PLUGIN" ] || { echo "$PLUGIN_FILE was not produced" >&2; exit 1; }

STAGE="$BUILD_DIR/stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -a "$PLUGIN" "$STAGE/"

GGML_VERSION=$(sed -nE 's/^set\(GGML_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)\)$/\2/p' "$SRC/ggml/CMakeLists.txt" | paste -sd.)
[[ "$GGML_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "could not read the ggml version from $SRC/ggml/CMakeLists.txt" >&2; exit 1; }
printf '%s\n' "$GGML_VERSION" > "$STAGE/ggml-version"

if [ "$OS" = linux ]; then
  strip --strip-unneeded "$STAGE/$PLUGIN_FILE" 2>/dev/null || true
  NEEDED=$(readelf -d "$STAGE/$PLUGIN_FILE" | sed -nE 's/.*\(NEEDED\).*\[(.*)\]/\1/p')
  echo "$NEEDED" | grep -qx 'libvulkan.so.1' || { echo "$PLUGIN_FILE does not link libvulkan.so.1" >&2; exit 1; }
  for lib in $NEEDED; do
    case "$lib" in
      libggml-base.so.0|libvulkan.so.1|libstdc++.so.6|libm.so.6|libgcc_s.so.1|libc.so.6|libgomp.so.1) ;;
      *) echo "$PLUGIN_FILE needs $lib, which a user machine may not have" >&2; exit 1 ;;
    esac
  done
  RUNPATH=$(readelf -d "$STAGE/$PLUGIN_FILE" | sed -nE 's/.*R(UN)?PATH.*\[(.*)\]/\2/p')
  [ "$RUNPATH" = '$ORIGIN' ] || { echo "$PLUGIN_FILE has run path '$RUNPATH', expected \$ORIGIN" >&2; exit 1; }
fi

for f in "$STAGE"/*; do
  base=$(basename "$f")
  case "$base" in
    "$PLUGIN_FILE"|ggml-version) ;;
    *) echo "unexpected file in the archive: $base" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT"
NAME="lune-voice-gpu-vulkan-${OS}-x86_64"
tar -czf "$OUT/$NAME.tar.gz" -C "$STAGE" .
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )

echo
echo "  $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
cat "$OUT/$NAME.tar.gz.sha256"
echo
echo "Upload both files to the GitHub release, then record the sha256 and the"
echo "byte size in VULKAN_$(echo "$OS" | tr '[:lower:]' '[:upper:]')_X86_64 (crates/lune-persist/src/gpu_plugin.rs)."
