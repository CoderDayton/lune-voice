#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# End-to-end test of the updater's install path, entirely on this machine.
#
#   cgs updater-test          # build both AppImages, serve, print instructions
#   cgs updater-test --env KEEP=1   # reuse AppImages already in the workdir
#
# The unit tests in desktop-ui/src-tauri/src/updater_tests.rs cover check +
# download + signature verification. What they cannot cover is the install
# itself — replacing the running app and restarting it. This does that for
# real, against a local endpoint.
#
# Both AppImages are built with `--debug` on purpose: release builds refuse a
# plain-HTTP endpoint, and serving HTTPS locally would mean a self-signed cert
# and loosening the shipped config. Debug builds allow HTTP while still
# verifying the signature against the pinned public key, so the trust path
# under test is the real one.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${TMPDIR:-/tmp}/lune-updater-test"
PORT="${PORT:-8477}"
CONF="$REPO_ROOT/desktop-ui/src-tauri/tauri.conf.json"
KEY_PATH="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.tauri/lune-voice-updater.key}"
BUNDLE_DIR="$REPO_ROOT/target/debug/bundle/appimage"

[ -f "$KEY_PATH" ] || { echo "no signing key at $KEY_PATH" >&2; exit 1; }
mkdir -p "$WORK_DIR"

# `createUpdaterArtifacts` + a pinned `pubkey` make the bundler demand the
# matching private key; without these exported it aborts before bundling.
export TAURI_SIGNING_PRIVATE_KEY_PATH="$KEY_PATH"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}"

# The version bump below rewrites a tracked file. Restore it byte for byte —
# re-serialising the JSON would leave unrelated reformatting behind (non-ASCII
# escaped, compact arrays exploded) even when the version is put back.
CONF_BACKUP="$WORK_DIR/tauri.conf.json.orig"
cp "$CONF" "$CONF_BACKUP"
restore_version() { cp "$CONF_BACKUP" "$CONF"; }
trap restore_version EXIT

current_version() { python3 -c "import json;print(json.load(open('$CONF'))['version'])"; }
set_version() { python3 - "$1" <<PY
import json, sys
path = "$CONF"
conf = json.load(open(path))
conf["version"] = sys.argv[1]
open(path, "w", encoding="utf-8").write(
    json.dumps(conf, indent=2, ensure_ascii=False) + "\n"
)
PY
}

BASE_VERSION="$(current_version)"
NEXT_VERSION="99.0.0"

build_appimage() {
    (cd "$REPO_ROOT/desktop-ui" && bun tauri build --debug --features whisper-cpp --bundles appimage)
}

if [ "${KEEP:-}" != "1" ]; then
    echo "==> building $BASE_VERSION (the 'old' app)"
    build_appimage
    cp "$BUNDLE_DIR"/*.AppImage "$WORK_DIR/old.AppImage"
    chmod +x "$WORK_DIR/old.AppImage"

    echo "==> building $NEXT_VERSION (the update)"
    set_version "$NEXT_VERSION"
    build_appimage
    restore_version
    # Pack and sign here rather than via `createUpdaterArtifacts`: turning that
    # on would make every ordinary `cgs build` demand the private key. Same
    # shape the plugin expects — a .tar.gz with the .AppImage inside.
    cp "$BUNDLE_DIR"/*.AppImage "$WORK_DIR/new.AppImage"
    tar czf "$WORK_DIR/new.AppImage.tar.gz" -C "$WORK_DIR" new.AppImage
    rm -f "$WORK_DIR/new.AppImage" "$WORK_DIR/new.AppImage.tar.gz.sig"
    bunx tauri signer sign -f "$KEY_PATH" "$WORK_DIR/new.AppImage.tar.gz" >/dev/null
fi

if [ ! -s "$WORK_DIR/new.AppImage.tar.gz.sig" ]; then
    echo "no signature for the update payload — signing failed" >&2
    exit 1
fi

cat > "$WORK_DIR/latest.json" <<EOF
{
  "version": "$NEXT_VERSION",
  "pub_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "platforms": {
    "linux-x86_64-appimage": {
      "signature": "$(cat "$WORK_DIR/new.AppImage.tar.gz.sig")",
      "url": "http://127.0.0.1:$PORT/new.AppImage.tar.gz"
    }
  }
}
EOF
python3 -m json.tool "$WORK_DIR/latest.json" >/dev/null

echo
echo "serving $WORK_DIR on port $PORT — Ctrl-C to stop"
echo
echo "In another terminal, run the OLD app against this endpoint:"
echo
echo "  LUNE_UPDATER_ENDPOINT=http://127.0.0.1:$PORT/latest.json $WORK_DIR/old.AppImage"
echo
echo "Then click the tray icon -> Check for Updates…"
echo "Expected: a dialog offering $NEXT_VERSION (you have $BASE_VERSION), then the"
echo "app downloads, swaps itself and restarts. Confirm with:"
echo
echo "  ./old.AppImage --version    # or check the tray dialog again — it should"
echo "                              # now report $NEXT_VERSION as the latest"
echo
cd "$WORK_DIR" && python3 -m http.server "$PORT" --bind 127.0.0.1
