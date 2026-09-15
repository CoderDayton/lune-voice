#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Install (or remove) a user-level desktop entry for a locally built binary.
#
# Why this exists: the shell shows the icon of the *application* it matched a
# window to, and that match runs through an installed desktop entry — keyed on
# WM_CLASS (X11) or app_id (Wayland), both of which GTK derives from the binary
# name. Running straight out of `target/` installs no entry, so the window falls
# back to a generic icon while the tray (passed to appindicator as bytes over
# DBus) looks fine. Installing the .deb fixes it for real users; this covers the
# build-tree workflow.
#
#   cgs desktop-entry                     # install, pointing at target/release
#   cgs desktop-entry --env PROFILE=debug # point at target/debug instead
#   cgs desktop-entry --env REMOVE=1      # uninstall
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"

# The entry's basename and StartupWMClass must both equal the app identity.
# Tauri only sets a GTK application id when `app.enableGtkAppId` is true (it is
# not), so GTK falls back to g_get_prgname() — the binary name.
APP_ID="lune-voice"
DESKTOP_FILE="$APP_DIR/$APP_ID.desktop"

if [ "${REMOVE:-}" = "1" ]; then
    rm -f "$DESKTOP_FILE"
    for size in 32x32 128x128 256x256; do
        rm -f "$ICON_ROOT/$size/apps/$APP_ID.png"
    done
    echo "removed $DESKTOP_FILE and its icons"
else
    PROFILE="${PROFILE:-release}"
    BIN="$REPO_ROOT/target/$PROFILE/$APP_ID"
    if [ ! -x "$BIN" ]; then
        echo "no binary at $BIN — run \`cgs build\` first (or --env PROFILE=debug)" >&2
        exit 1
    fi

    # hicolor directory names must match the real pixel size, so 128x128@2x
    # (256px) is installed as 256x256.
    mkdir -p "$APP_DIR"
    while read -r src size; do
        mkdir -p "$ICON_ROOT/$size/apps"
        cp "$REPO_ROOT/desktop-ui/src-tauri/icons/$src" "$ICON_ROOT/$size/apps/$APP_ID.png"
    done <<EOF
32x32.png 32x32
128x128.png 128x128
128x128@2x.png 256x256
EOF

    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Lune Voice
Comment=Offline voice dictation powered by local AI
Exec=$BIN
Icon=$APP_ID
StartupWMClass=$APP_ID
Terminal=false
Categories=Utility;
EOF
    echo "installed $DESKTOP_FILE -> $BIN"
fi

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APP_DIR" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -f -t "$ICON_ROOT" >/dev/null 2>&1 || true
exit 0
