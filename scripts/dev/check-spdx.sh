#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Fail when a source file is missing its SPDX licence line.
#
# Checks tracked and untracked (not ignored) .rs .ts .tsx .css .sh .py .html
# files. The line must sit in the first three lines, which leaves room for a
# shebang or `<!doctype html>`. Skipped: vendor/, which keeps its upstream
# licence, and generated/, which ts-rs rewrites on every `cargo test -p lune-ui`.
#
# Usage: check-spdx.sh [repo-root]   (default: the repo holding this script)
set -euo pipefail

root="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
missing=0

while IFS= read -r -d '' f; do
    # A tracked file deleted in the working tree has nothing to check.
    [ -f "$root/$f" ] || continue
    if ! head -n 3 "$root/$f" | grep -q 'SPDX-License-Identifier: GPL-3.0-or-later'; then
        echo "missing SPDX line: $f" >&2
        missing=$((missing + 1))
    fi
done < <(git -C "$root" ls-files -z --cached --others --exclude-standard -- \
    '*.rs' '*.ts' '*.tsx' '*.css' '*.sh' '*.py' '*.html' \
    ':!vendor/*' ':!*/generated/*')

if [ "$missing" -gt 0 ]; then
    echo "$missing file(s) need an 'SPDX-License-Identifier: GPL-3.0-or-later' comment in their first lines" >&2
    exit 1
fi
