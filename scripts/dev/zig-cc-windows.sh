#!/bin/sh
# C/C++ cross-compiler for the `x86_64-pc-windows-gnu` check, backed by `zig cc`.
#
# `cc-rs` passes the Rust triple as `--target=x86_64-pc-windows-gnu`. Zig's
# target parser rejects that spelling with `UnknownOperatingSystem` (it wants
# `x86_64-windows-gnu`), and the `-target` it does accept must not appear twice.
# So every target flag is dropped here and one correct flag is supplied.
#
# Wired in via the `[env]` block in `.cargo/config.toml`; `cgs check-windows`
# runs the whole check.
set -e

# Rebuilt in the positional parameters rather than a string: an argument goes
# through untouched that way. Accumulating into one string and `eval`-ing it
# would let the shell re-expand `$`, backticks and quotes inside a flag —
# `-DVERSION="1.0"` arrives as `-DVERSION=1.0`, and a `$` in a path is expanded
# away — silently miscompiling instead of failing.
first=1
skip_next=0
for a in "$@"; do
    if [ "$skip_next" -eq 1 ]; then
        skip_next=0
        continue
    fi
    case "$a" in
        --target=*) continue ;;
        -target)
            skip_next=1
            continue
            ;;
    esac
    if [ "$first" -eq 1 ]; then
        set -- "$a"
        first=0
    else
        set -- "$@" "$a"
    fi
done
# Nothing survived the filter, so drop the originals `$@` still holds.
if [ "$first" -eq 1 ]; then
    set --
fi

exec zig cc -target x86_64-windows-gnu "$@"
