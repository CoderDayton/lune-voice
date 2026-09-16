// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Session length as `m:ss`. Minutes are not folded into hours, so a long
 * session reads `61:01` rather than looking like a one-minute one.
 */
export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/**
 * Suggested file name for an exported transcript, stamped with the session's
 * local start time to the minute: `lune-YYYY-MM-DD-HHMM.txt`.
 */
export function transcriptFileName(startedAt: string): string {
  const d = new Date(startedAt);
  const pad = (n: number) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
  return `lune-${stamp}.txt`;
}
