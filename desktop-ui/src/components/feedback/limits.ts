// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The feedback contract shared with `src-tauri/src/feedback.rs`.
 *
 * The command is the authority — it re-checks every one of these. The dialog
 * only needs them to gate the form before spending a submission and to render
 * an honest character counter. They are hand-mirrored, so
 * `tests/unit/feedback-limits.test.ts` parses the Rust source and fails when
 * the two sides drift; without it a change on either side would surface as a
 * client-side gate the command rejects.
 */

/** `MAX_MESSAGE_CHARS` in `feedback.rs`. */
export const MAX_MESSAGE_CHARS = 1800;
/** `MIN_MESSAGE_CHARS` in `feedback.rs`. */
export const MIN_MESSAGE_CHARS = 10;
/** `COOLDOWN` in `feedback.rs`, in seconds. */
export const COOLDOWN_SECONDS = 120;

/** `CATEGORIES` in `feedback.rs`. Sent verbatim as the bold header of the
 *  relayed message, so these read as channel labels, not UI copy. */
export const CATEGORIES = [
  'Bug',
  'Crash',
  'Accuracy',
  'Performance',
  'Idea',
  'Question',
  'Feedback',
] as const;

export type Category = (typeof CATEGORIES)[number];

/** Code points, matching the backend's `chars().count()`. A JS `.length` counts
 *  UTF-16 units, so five emoji would pass a gate the command then rejects. */
export function charCount(value: string): number {
  return [...value].length;
}
