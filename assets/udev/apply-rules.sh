#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger --subsystem-match=misc --sysname-match=uinput 2>/dev/null || true
udevadm trigger --subsystem-match=input --property-match=ID_INPUT_KEYBOARD=1 2>/dev/null || true
exit 0
