#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Static archiver paired with `zig-cc-windows.sh`. `cc-rs` looks for a
# target-prefixed `ar` and finds no `x86_64-w64-mingw32-ar` on a Linux box.
exec zig ar "$@"
