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

bold "==> Done."
