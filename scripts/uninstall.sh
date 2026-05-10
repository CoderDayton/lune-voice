#!/usr/bin/env bash
set -euo pipefail

BINARY="lune-voice"

# ── Colors ────────────────────────────────────────────────────────────────────
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
info()  { printf '  %s\n' "$*"; }

bold "==> lune-voice uninstall"

# ── Step 1: Shut down running daemon ─────────────────────────────────────────
if command -v "$BINARY" &>/dev/null; then
    if "$BINARY" status &>/dev/null 2>&1; then
        info "Daemon is running — sending quit..."
        "$BINARY" quit || true
        sleep 1
        if "$BINARY" status &>/dev/null 2>&1; then
            red "Daemon did not stop; killing by D-Bus name..."
            if command -v busctl &>/dev/null; then
                pid=$(busctl status com.lunevoice.Daemon1 2>/dev/null \
                    | awk '/^PID:/ { print $2 }') || true
                if [[ -n "${pid:-}" ]]; then
                    info "Killing PID $pid"
                    kill "$pid" || true
                    sleep 1
                fi
            fi
        else
            green "Daemon stopped."
        fi
    else
        info "Daemon not running."
    fi

    # ── Step 2: Remove binary ─────────────────────────────────────────────────
    info "Removing binary..."
    cargo uninstall "$BINARY" 2>/dev/null || rm -f "$(command -v "$BINARY")"
    green "Removed."
else
    info "lune-voice is not installed."
fi

# ── Step 3: Remove ORT env snippet + shell rc references ────────────────────
PROFILE_SNIPPET="$HOME/.config/lune-voice/ort-env.sh"
if [[ -f "$PROFILE_SNIPPET" ]]; then
    info "Removing ORT env snippet: $PROFILE_SNIPPET"
    rm -f "$PROFILE_SNIPPET"
fi

for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
    if [[ -f "$rc" ]] && grep -qF '# lune-voice ORT CUDA libs' "$rc" 2>/dev/null; then
        info "Cleaning ORT env source from $(basename "$rc")"
        # Drop the "# lune-voice ORT CUDA libs" marker comment and the
        # source line directly after it, leaving the rest untouched.
        awk '
          $0 == "# lune-voice ORT CUDA libs" { skip = 1; next }
          skip { skip = 0; next }
          { print }
        ' "$rc" > "$rc.lune-tmp" && mv "$rc.lune-tmp" "$rc"
    fi
done

# ── Step 4: Remove model symlink (only if it IS a symlink) ──────────────────
XDG_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_MODELS="$XDG_DATA/lune-voice/models"
if [[ -L "$APP_MODELS" ]]; then
    info "Removing model symlink: $APP_MODELS"
    rm "$APP_MODELS"
elif [[ -d "$APP_MODELS" ]]; then
    info "Models directory at $APP_MODELS is not a symlink — leaving in place."
    info "  Remove manually if desired: rm -rf '$APP_MODELS'"
fi

bold "==> Done."
