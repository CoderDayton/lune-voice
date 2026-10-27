# SPDX-License-Identifier: GPL-3.0-or-later
# Sourced by cgs dev/build/bench. Sets BACKEND (cpu|vulkan) and FEATURES.
# whisper-vulkan fails to build if any one of glslc, the Vulkan headers or
# SPIRV-Headers is missing, so all three are required for auto-pick.
if [ -z "${BACKEND:-}" ]; then
  if command -v glslc >/dev/null 2>&1 && pkg-config --exists vulkan SPIRV-Headers 2>/dev/null; then
    BACKEND=vulkan
  else
    BACKEND=cpu
  fi
fi
case "$BACKEND" in
  cpu)    FEATURES="whisper-cpp" ;;
  vulkan) FEATURES="whisper-vulkan" ;;
  *)      echo "BACKEND must be cpu|vulkan (macOS Metal: see docs/RELEASE.md)"; exit 1 ;;
esac
echo "GPU backend: $BACKEND (override with --env BACKEND=cpu|vulkan)"
