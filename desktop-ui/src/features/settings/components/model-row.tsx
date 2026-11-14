// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { AlertTriangle, Check, Download, RotateCw, Trash2, X } from 'lucide-react';
import { formatBytes } from '../../../lib/format-bytes';

/** Where one model stands, with what the row needs to draw it. */
export type RowState =
  | { kind: 'available' }
  | { kind: 'queued' }
  | { kind: 'downloading'; done: number; total: number }
  | { kind: 'failed'; error: string }
  | { kind: 'installed' };

interface ModelRowProps {
  label: string;
  description: string;
  size: string;
  state: RowState;
  inUse: boolean;
  onDownload: () => void;
  /** Takes the model off the queue, or stops its transfer. */
  onCancel: () => void;
  onRetry: () => void;
  onDelete: () => void;
}

/**
 * One model: name, size and note on the left, and on the right the one thing
 * that can be done with it now. A download shows its progress in that same
 * spot, so the row never changes height or jumps.
 */
export function ModelRow({
  label,
  description,
  size,
  state,
  inUse,
  onDownload,
  onCancel,
  onRetry,
  onDelete,
}: ModelRowProps) {
  const { t } = useLingui();
  return (
    <div className="model-row" data-in-use={inUse || undefined}>
      <div className="flex min-w-0 flex-col gap-0.5">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-base font-medium text-content-strong">{label}</span>
          <span className="text-sm tabular-nums text-content-muted">{size}</span>
          {inUse && <span className="text-xs font-medium text-primary">{t`In use`}</span>}
        </div>
        <p className="text-sm text-content-muted">{description}</p>
        {state.kind === 'failed' && (
          <p role="alert" className="flex items-start gap-1.5 text-sm text-error select-text">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" />
            <span className="min-w-0 break-words">{state.error}</span>
          </p>
        )}
      </div>
      <div className="model-row-action">
        <RowAction
          label={label}
          state={state}
          onDownload={onDownload}
          onCancel={onCancel}
          onRetry={onRetry}
          onDelete={onDelete}
        />
      </div>
    </div>
  );
}

function RowAction({
  label,
  state,
  onDownload,
  onCancel,
  onRetry,
  onDelete,
}: Pick<ModelRowProps, 'label' | 'state' | 'onDownload' | 'onCancel' | 'onRetry' | 'onDelete'>) {
  const { t } = useLingui();
  const cancel = (
    <button
      type="button"
      className="btn btn-ghost btn-sm btn-square text-content-muted hover:text-content-strong"
      onClick={onCancel}
      aria-label={t`Cancel download of ${label}`}
      title={t`Cancel`}
    >
      <X size={14} />
    </button>
  );

  switch (state.kind) {
    case 'available':
      return (
        <button
          type="button"
          className="btn btn-soft btn-sm gap-1.5"
          onClick={onDownload}
          aria-label={t`Download ${label}`}
        >
          <Download size={14} />
          {t`Download`}
        </button>
      );
    case 'queued':
      return (
        <>
          <span className="text-sm text-content-muted">{t`Queued`}</span>
          {cancel}
        </>
      );
    case 'downloading': {
      const pct = state.total > 0 ? Math.round((state.done / state.total) * 100) : 0;
      return (
        <>
          <div className="flex w-36 flex-col gap-1">
            <progress
              className="progress progress-primary h-1.5 w-full"
              value={state.done}
              // `max={0}` renders an indeterminate bar; keep it positive.
              max={state.total || 1}
              aria-label={t`Download progress of ${label}`}
            />
            <div className="flex justify-between text-xs tabular-nums text-content-subtle">
              <span>
                {formatBytes(state.done)} / {formatBytes(state.total)}
              </span>
              <span className="font-medium text-primary">{pct}%</span>
            </div>
          </div>
          {cancel}
        </>
      );
    }
    case 'failed':
      return (
        <>
          <button type="button" className="btn btn-soft btn-sm gap-1.5" onClick={onRetry}>
            <RotateCw size={14} />
            {t`Retry`}
          </button>
          {cancel}
        </>
      );
    case 'installed':
      return (
        <>
          <span className="inline-flex items-center gap-1 text-sm font-medium text-success">
            <Check size={14} strokeWidth={2.5} />
            {t`Installed`}
          </span>
          <button
            type="button"
            className="btn btn-ghost btn-sm btn-square text-content-muted hover:text-error"
            onClick={onDelete}
            aria-label={t`Delete ${label}`}
            title={t`Delete`}
          >
            <Trash2 size={14} />
          </button>
        </>
      );
  }
}
