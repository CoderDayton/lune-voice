// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, CircleAlert, Loader2, Mic, Zap } from 'lucide-react';
import { ByteProgress } from '../../../components/ui/byte-progress';
import { formatBytes } from '../../../lib/format-bytes';
import type { ChecklistRow } from '../setup-logic';
import type { SetupChecklist } from '../use-model-setup';

interface DownloadChecklistProps {
  checklist: SetupChecklist;
  /** The GPU finished loading into the running app. */
  gpuActive: boolean;
  gpuError: string | null;
}

/**
 * Every download of a setup run in one list, in the order they happen. One row
 * is active at a time; the bar on top counts bytes across all of them, so the
 * user can see both what is happening now and how much is left.
 */
export function DownloadChecklist({ checklist, gpuActive, gpuError }: DownloadChecklistProps) {
  const { t } = useLingui();
  const { rows, doneBytes, totalBytes, settled } = checklist;
  const pct = totalBytes > 0 ? Math.round((doneBytes / totalBytes) * 100) : 0;

  return (
    <div className="setup-card p-4">
      {/* Only while something is moving: once every row has its outcome the
          bar says nothing the rows don't. */}
      {!settled && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs text-content-subtle">
            <span className="font-medium uppercase tracking-wider">{t`Overall progress`}</span>
            {totalBytes > 0 && (
              <span className="tabular-nums">
                {formatBytes(doneBytes)} / {formatBytes(totalBytes)} · {pct}%
              </span>
            )}
          </div>
          <progress
            className="progress progress-primary mt-2 h-2 w-full"
            value={doneBytes}
            max={totalBytes || 1}
            aria-label={t`Overall progress`}
          />
        </div>
      )}
      <ul className="flex flex-col divide-y divide-surface-border">
        {rows.map((row) => (
          <Row key={row.id} row={row} gpuActive={gpuActive} gpuError={gpuError} />
        ))}
      </ul>
    </div>
  );
}

function Row({
  row,
  gpuActive,
  gpuError,
}: {
  row: ChecklistRow;
  gpuActive: boolean;
  gpuError: string | null;
}) {
  const { t } = useLingui();
  const gpu = row.id === 'gpu';
  const Icon = gpu ? Zap : Mic;

  const tile =
    row.state === 'done'
      ? 'bg-success/15 text-success'
      : row.state === 'failed'
        ? 'bg-warning/15 text-warning'
        : row.state === 'queued'
          ? 'bg-surface-wash text-content-subtle'
          : 'bg-primary/15 text-primary';

  const icon =
    row.state === 'done' ? (
      <Check size={18} strokeWidth={2.5} />
    ) : row.state === 'failed' ? (
      <CircleAlert size={18} strokeWidth={1.8} />
    ) : row.state === 'active' || row.state === 'activating' ? (
      <Loader2 size={18} className="animate-spin" />
    ) : (
      <Icon size={18} strokeWidth={1.6} />
    );

  const status =
    row.state === 'queued'
      ? t`Waiting`
      : row.state === 'activating'
        ? t`Turning on your GPU…`
        : row.state === 'failed'
          ? gpu
            ? (gpuError ??
              t`Could not download GPU support — transcription will run on the CPU. You can retry under Settings › Models.`)
            : // The alert below the list carries the error itself.
              t`Interrupted`
          : gpu
            ? gpuActive
              ? t`Transcription now runs on your GPU.`
              : t`Restart Lune Voice to start using your GPU.`
            : t`Downloaded`;

  return (
    <li
      className="flex items-start gap-3 py-3 first:pt-0 last:pb-0"
      aria-current={row.state === 'active' ? 'step' : undefined}
    >
      <span className={`setup-tile ${tile}`}>{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <p className="truncate text-sm font-medium text-content-strong">
            {gpu ? t`GPU support` : row.name}
          </p>
          {row.totalBytes > 0 && row.state !== 'active' && (
            <span className="shrink-0 text-xs tabular-nums text-content-subtle">
              {formatBytes(row.totalBytes)}
            </span>
          )}
        </div>
        {row.state === 'active' ? (
          <ByteProgress downloadedBytes={row.doneBytes} totalBytes={row.totalBytes} />
        ) : (
          // `select-text` because a failure can carry the backend's own
          // message, and the shell is `select-none`.
          <p className="mt-0.5 text-xs leading-snug text-content-subtle select-text">{status}</p>
        )}
      </div>
    </li>
  );
}
