#!/usr/bin/env bun
// SPDX-License-Identifier: GPL-3.0-or-later
// Auto-wrap user-facing string props in Lingui macros.
// Usage: bun scripts/wrap-i18n.ts [--dry-run] [glob...]

import { readFileSync, writeFileSync } from "fs";
import { glob } from "glob";

const dryRun = process.argv.includes("--dry-run");
const patterns = process.argv.filter((a) => !a.startsWith("-") && a !== process.argv[0] && a !== process.argv[1]);
if (patterns.length === 0) patterns.push("src/**/*.tsx");

// Props whose string values should be wrapped in t``
const I18N_PROPS = new Set([
  "label",
  "description",
  "title",
  "note",
  "kicker",
  "placeholder",
  "aria-label",
]);

// Skip these values — they're technical, not user-facing
function shouldSkip(value: string): boolean {
  if (value.length === 0) return true;
  // Single word with no spaces (likely a CSS class, enum, or key)
  if (!/\s/.test(value) && value.length < 30) return true;
  // Looks like a URL
  if (value.startsWith("http") || value.startsWith("/")) return true;
  // Looks like a CSS class
  if (value.includes("btn-") || value.includes("badge-")) return true;
  // Already contains template syntax
  if (value.includes("`") || value.includes("${")) return true;
  return false;
}

let totalFiles = 0;
let totalWrapped = 0;

for (const pattern of patterns) {
  const files = glob.sync(pattern, { cwd: process.cwd() });

  for (const file of files) {
    let source = readFileSync(file, "utf-8");
    let changed = false;
    let needsUseLingui = false;
    let wrappedCount = 0;

    // 1. Wrap string props: prop="some text" → prop={t`some text`}
    //    Matches: label="..." or description="..." etc.
    const propRegex = new RegExp(
      `((?:${[...I18N_PROPS].join("|")})=)"([^"]*)"`,
      "g",
    );

    source = source.replace(propRegex, (match, prefix: string, value: string) => {
      if (shouldSkip(value)) return match;
      needsUseLingui = true;
      wrappedCount++;
      return `${prefix}{t\`${value}\`}`;
    });

    // 2. Also handle single-quoted props
    const singleQuoteRegex = new RegExp(
      `((?:${[...I18N_PROPS].join("|")})=)'([^']*)'`,
      "g",
    );

    source = source.replace(singleQuoteRegex, (match, prefix: string, value: string) => {
      if (shouldSkip(value)) return match;
      needsUseLingui = true;
      wrappedCount++;
      return `${prefix}{t\`${value}\`}`;
    });

    if (wrappedCount > 0) changed = true;

    // 3. Add imports if needed
    if (needsUseLingui && !source.includes("useLingui")) {
      // Find the end of the last top-level import block (handles multi-line imports)
      let insertPos = 0;
      const importRe = /^import\s[\s\S]*?from\s+["'][^"']+["'];?\s*$/gm;
      let m: RegExpExecArray | null;
      while ((m = importRe.exec(source)) !== null) {
        insertPos = m.index + m[0].length;
      }
      if (insertPos > 0) {
        source =
          source.slice(0, insertPos) +
          '\nimport { useLingui } from "@lingui/react/macro";' +
          source.slice(insertPos);
        changed = true;
      }
    }

    // 4. Add `const { t } = useLingui();` to components that need it
    if (needsUseLingui && !source.includes("const { t }")) {
      // Find the first line after "function ComponentName(" that starts with "  "
      // or "  const " — insert before it
      const funcMatch = source.match(/export function \w+\([^)]*\)\s*\{/);
      if (funcMatch && funcMatch.index !== undefined) {
        const bracePos = source.indexOf("{", funcMatch.index + "export function ".length);
        const nextLine = source.indexOf("\n", bracePos);
        source =
          source.slice(0, nextLine + 1) +
          "  const { t } = useLingui();\n" +
          source.slice(nextLine + 1);
        changed = true;
      }
    }

    if (changed) {
      totalFiles++;
      totalWrapped += wrappedCount;
      if (dryRun) {
        console.log(`[dry-run] ${file}: ${wrappedCount} props to wrap`);
      } else {
        writeFileSync(file, source);
        console.log(`[wrapped] ${file}: ${wrappedCount} props`);
      }
    }
  }
}

console.log(`\nDone: ${totalWrapped} props across ${totalFiles} files${dryRun ? " (dry run)" : ""}`);
