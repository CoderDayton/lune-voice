// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { formatBytes } from '../../lib/format-bytes';

interface ByteProgressProps {
  downloadedBytes: number;
  /** Content-Length reported by the server; 0 when it sent none. */
  totalBytes: number;
  /** Published archive size, used when the server sent no Content-Length. */
  fallbackTotalBytes?: number;
}

/**
 * Byte-count progress bar for a download in flight.
 *
 * Shared so every download surface reads the same way and agrees on what to do
 * when the server omits Content-Length.
 */
export function ByteProgress({
  downloadedBytes,
  totalBytes,
  fallbackTotalBytes = 0,
}: ByteProgressProps) {
  const { t } = useLingui();
  const total = totalBytes || fallbackTotalBytes;
  const pct = total > 0 ? Math.round((downloadedBytes / total) * 100) : 0;

  return (
    <div className="mt-2 space-y-1">
      <progress
        className="progress progress-primary h-1.5 w-full"
        value={downloadedBytes}
        // `max={0}` is invalid and renders as an indeterminate bar; keep a
        // positive maximum so the element stays a well-formed 0% bar instead.
        max={total || 1}
        aria-label={t`Download progress`}
      />
      <div className="flex items-center justify-between text-xs tabular-nums text-content-subtle">
        <span>
          {formatBytes(downloadedBytes)} / {formatBytes(total)}
        </span>
        <span className="ml-2 shrink-0 font-medium text-primary">{pct}%</span>
      </div>
    </div>
  );
}
