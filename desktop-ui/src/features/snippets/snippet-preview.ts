// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Placeholder substitution for the snippet composer's live preview.
 *
 * Its whole job is to agree with `lune-dict::snippets::render_body`, which is
 * what actually runs at expansion time. A preview that disagrees with the
 * expansion is worse than no preview, so this lives on its own and is tested
 * against the cases where the two could drift.
 */

/**
 * Placeholders the backend substitutes at expansion time. Known tokens are
 * replaced; anything else in braces passes through verbatim.
 */
export const PLACEHOLDERS = ['date', 'time', 'datetime', 'clipboard'] as const;

/** Fill placeholders in `body` the way `render_body` will. */
export function previewBody(body: string, clipboardLabel: string): string {
  const now = new Date();
  const date = now.toLocaleDateString(undefined, {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
  const time = now.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
  const values: Record<string, string> = {
    date,
    time,
    datetime: `${date} at ${time}`,
    clipboard: clipboardLabel,
  };

  // Deliberately a scan rather than a regex. `render_body` takes the first `{`
  // and then the next `}`, so a token may itself contain `{`: "{a{date}" is the
  // unknown token "a{date" and survives verbatim. A `\{([^{}]*)\}` pattern
  // would instead find "{date}" nested inside it and substitute, previewing
  // text the expansion never produces.
  let out = '';
  let rest = body;
  for (;;) {
    const open = rest.indexOf('{');
    if (open === -1) break;
    out += rest.slice(0, open);
    const after = rest.slice(open + 1);
    const close = after.indexOf('}');
    // Unclosed brace: the remainder is literal, exactly as the backend leaves it.
    if (close === -1) return out + rest.slice(open);
    const token = after.slice(0, close);
    out += values[token.toLowerCase()] ?? `{${token}}`;
    rest = after.slice(close + 1);
  }
  return out + rest;
}
