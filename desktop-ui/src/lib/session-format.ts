// SPDX-License-Identifier: GPL-3.0-or-later

import type { I18n } from '@lingui/core';
import { msg } from '@lingui/core/macro';

const FAILED_PREFIX = 'Failed: ';

export function sessionStatusLabel(i18n: I18n, status: string): string {
  if (status === 'Completed') return i18n._(msg`Completed`);
  if (status === 'Cancelled') return i18n._(msg`Cancelled`);
  if (status === 'Active') return i18n._(msg`In progress`);
  if (status.startsWith(FAILED_PREFIX)) {
    const reason = status.slice(FAILED_PREFIX.length);
    return i18n._(msg`Failed: ${reason}`);
  }
  return status;
}

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
