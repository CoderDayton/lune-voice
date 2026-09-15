#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Windows resource compiler for the `x86_64-pc-windows-gnu` check.
#
# `embed-resource` (via `tauri-winres`, via `tauri_build::build()`) compiles the
# icon/manifest resource for `lune-ui`. It picks `x86_64-w64-mingw32-windres`
# from binutils-mingw-w64, which preprocesses the `.rc` by shelling out to
# `x86_64-w64-mingw32-gcc` — a MinGW *compiler*, which a zig-only box does not
# have. `zig cc` stands in for that step here, matching the rest of the
# cross-check toolchain.
#
# The preprocessor arguments replace windres' own defaults, so `-E -xc` and
# `-DRC_INVOKED` have to be repeated verbatim: `.rc` files are C-preprocessed
# and the resource headers key off that define.
#
# Wired in as `RC_x86_64_pc_windows_gnu` by the `[env]` block in
# `.cargo/config.toml`; `cgs check-windows` runs the whole check.
set -e

exec x86_64-w64-mingw32-windres \
    --preprocessor=zig \
    --preprocessor-arg=cc \
    --preprocessor-arg=-target \
    --preprocessor-arg=x86_64-windows-gnu \
    --preprocessor-arg=-E \
    --preprocessor-arg=-xc \
    --preprocessor-arg=-DRC_INVOKED \
    "$@"
