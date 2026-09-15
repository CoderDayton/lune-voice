// SPDX-License-Identifier: GPL-3.0-or-later

/** Binary step between KB, MB and GB — the units disk and download sizes use. */
const UNIT = 1024;
const MB = UNIT * UNIT;
const GB = MB * UNIT;

/** Human-readable byte size, e.g. `192 MB` / `1.2 GB`. */
export function formatBytes(bytes: number): string {
  // Nothing is nothing: a download that has not started reads `0 KB`, while a
  // real but sub-kilobyte file still rounds up rather than vanishing.
  if (bytes <= 0) return '0 KB';
  if (bytes < MB) return `${Math.max(1, Math.round(bytes / UNIT))} KB`;
  if (bytes < GB) return `${Math.round(bytes / MB)} MB`;
  return `${(bytes / GB).toFixed(1)} GB`;
}
