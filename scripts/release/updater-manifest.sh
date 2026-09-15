#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build the `latest.json` the updater plugin fetches, from artifacts already in
# release-artifacts/.
#
#   bash scripts/release/updater-manifest.sh v0.2.0
#
# The plugin refuses bytes that don't match how the app was installed, so a
# single URL per platform cannot serve deb, rpm and AppImage users. This emits
# one key per Linux format and signs the deb and rpm, which tauri-bundler does
# NOT sign itself (it only produces updater artifacts for AppImage on Linux).
#
# Windows has one installer and so one key, `windows-x86_64` — the bare key the
# plugin falls back to when no per-format variant matches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACTS_DIR="$REPO_ROOT/release-artifacts"
GITHUB_REPO="CoderDayton/lune-voice"
KEY_PATH="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.tauri/lune-voice-updater.key}"

TAG="${1:-}"
if [ -z "$TAG" ]; then
    echo "usage: $0 <release-tag>   e.g. $0 v0.2.0" >&2
    exit 1
fi
VERSION="${TAG#v}"

# The plugin compares the manifest version against the version the installed
# binary reports. If the tag and the built version disagree, every client
# re-offers the same update forever after installing it.
CONF_VERSION="$(python3 -c "import json;print(json.load(open('$REPO_ROOT/desktop-ui/src-tauri/tauri.conf.json'))['version'])")"
if [ "$VERSION" != "$CONF_VERSION" ]; then
    echo "tag $TAG implies version $VERSION but tauri.conf.json says $CONF_VERSION" >&2
    echo "bump the config (and rebuild) or tag the version that was built" >&2
    exit 1
fi

if [ ! -f "$KEY_PATH" ]; then
    echo "no signing key at $KEY_PATH — set TAURI_SIGNING_PRIVATE_KEY_PATH" >&2
    exit 1
fi

# GitHub replaces spaces with dots in uploaded asset filenames, so the download
# URL never matches the local name verbatim.
asset_url() {
    printf 'https://github.com/%s/releases/download/%s/%s\n' \
        "$GITHUB_REPO" "$TAG" "$(basename "$1" | tr ' ' '.')"
}

# Sign a file unless the bundler already produced a .sig next to it.
signature_for() {
    local file="$1"
    if [ ! -f "$file.sig" ]; then
        # The password env var must be set even when empty, or the CLI prompts
        # and a release run blocks forever on a TTY that isn't there.
        (cd "$REPO_ROOT/desktop-ui" && \
            TAURI_SIGNING_PRIVATE_KEY_PASSWORD="${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}" \
            bunx tauri signer sign -f "$KEY_PATH" "$file" >/dev/null)
    fi
    cat "$file.sig"
}

emit_platform() {
    local key="$1" file="$2" trailing="$3" signature
    # Assign separately: a command substitution used directly as a printf
    # argument swallows its exit status, so a failed signing would emit an
    # empty signature into a manifest that is still valid JSON and that every
    # client then rejects.
    signature="$(signature_for "$file")"
    if [ -z "$signature" ]; then
        echo "failed to obtain a signature for $file" >&2
        exit 1
    fi
    printf '    "%s": {\n      "signature": "%s",\n      "url": "%s"\n    }%s\n' \
        "$key" "$signature" "$(asset_url "$file")" "$trailing"
}

# The AppImage has to reach the client as a .tar.gz — the plugin extracts the
# archive and looks for the .AppImage inside it. tauri's `createUpdaterArtifacts`
# would build this during bundling, but enabling that makes EVERY build demand
# the private key (a pinned pubkey without one is a hard error in the CLI), which
# would break `cgs build` for anyone who doesn't hold it. Packing it here keeps
# signing in one place and builds key-free.
pack_appimage() {
    local appimage="$1"
    local archive="$appimage.tar.gz"
    [ -f "$archive" ] || tar czf "$archive" -C "$(dirname "$appimage")" "$(basename "$appimage")"
    printf '%s\n' "$archive"
}

shopt -s nullglob
RAW_APPIMAGE=("$ARTIFACTS_DIR"/*.AppImage)
APPIMAGE=("$ARTIFACTS_DIR"/*.AppImage.tar.gz)
DEB=("$ARTIFACTS_DIR"/*.deb)
RPM=("$ARTIFACTS_DIR"/*.rpm)
# `*-setup.exe` is the NSIS bundle; the pattern excludes any other .exe that
# happens to be lying in the directory.
NSIS=("$ARTIFACTS_DIR"/*-setup.exe)
# The portable archive starts with a version number, which keeps the GPU plugin
# archive (`lune-voice-gpu-...tar.gz`, ~500 MB) out of the app release.
PORTABLE=("$ARTIFACTS_DIR"/lune-voice-[0-9]*-linux-*.tar.gz)
shopt -u nullglob

if [ ${#APPIMAGE[@]} -eq 0 ] && [ ${#RAW_APPIMAGE[@]} -gt 0 ]; then
    APPIMAGE=("$(pack_appimage "${RAW_APPIMAGE[0]}")")
fi

if [ ${#APPIMAGE[@]} -eq 0 ] && [ ${#DEB[@]} -eq 0 ] && [ ${#RPM[@]} -eq 0 ] \
    && [ ${#NSIS[@]} -eq 0 ]; then
    echo "no artifacts in $ARTIFACTS_DIR — run \`cgs package\` first" >&2
    exit 1
fi

MANIFEST="$ARTIFACTS_DIR/latest.json"
# Build into a temp file and move it into place, so a signing failure part way
# through cannot leave a truncated latest.json for someone to upload.
MANIFEST_TMP="$(mktemp "$ARTIFACTS_DIR/latest.json.XXXXXX")"
trap 'rm -f "$MANIFEST_TMP"' EXIT
{
    printf '{\n  "version": "%s",\n  "pub_date": "%s",\n  "platforms": {\n' \
        "$VERSION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Track which entry is last so the JSON stays comma-correct.
    entries=()
    [ ${#APPIMAGE[@]} -gt 0 ] && entries+=("linux-x86_64-appimage:${APPIMAGE[0]}")
    [ ${#DEB[@]} -gt 0 ]      && entries+=("linux-x86_64-deb:${DEB[0]}")
    [ ${#RPM[@]} -gt 0 ]      && entries+=("linux-x86_64-rpm:${RPM[0]}")
    [ ${#NSIS[@]} -gt 0 ]     && entries+=("windows-x86_64:${NSIS[0]}")

    for i in "${!entries[@]}"; do
        entry="${entries[$i]}"
        trailing=","
        [ "$i" -eq $((${#entries[@]} - 1)) ] && trailing=""
        emit_platform "${entry%%:*}" "${entry#*:}" "$trailing"
    done

    printf '  }\n}\n'
} > "$MANIFEST_TMP"

# Separate statements: as the left side of `&&` a failure here is exempt from
# `set -e`, so an invalid manifest would exit 0 having printed nothing.
python3 -m json.tool "$MANIFEST_TMP" >/dev/null
mv "$MANIFEST_TMP" "$MANIFEST"
echo "wrote $MANIFEST"
echo "valid JSON"
grep -oE '"(linux|windows)-x86_64(-[a-z]*)?"' "$MANIFEST" | tr -d '"' | sed 's/^/  platform: /'
echo
echo "Upload latest.json together with the artifacts, e.g.:"
upload=("$MANIFEST" "${RAW_APPIMAGE[@]}" "${APPIMAGE[@]}" "${DEB[@]}" "${RPM[@]}" "${NSIS[@]}" "${PORTABLE[@]}")
printf '  gh release upload %q' "$TAG"
printf ' %q' "${upload[@]}"
echo
