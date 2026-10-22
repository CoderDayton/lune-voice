// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, Lock, type LucideIcon } from 'lucide-react';
import type { DownloadProgress } from '../use-model-setup';

interface ModelCardProps {
  name: string;
  description: string;
  required: boolean;
  icon: LucideIcon;
  ready: boolean;
  selected: boolean;
  active: boolean;
  canToggle: boolean;
  progress: DownloadProgress | null;
  onToggle: () => void;
}

export function ModelCard({
  name,
  description,
  required,
  icon: Icon,
  ready,
  selected,
  active,
  canToggle,
  progress,
  onToggle,
}: ModelCardProps) {
  const { t } = useLingui();
  const pct =
    progress && progress.fileCount > 0
      ? Math.round(((progress.fileIndex + 1) / progress.fileCount) * 100)
      : 0;

  // Card chrome. Only the two *transient* states tint the card — a download
  // in flight, and a pending selection. "Installed" does not: a green card
  // sitting in a stack of plain ones reads as a different kind of object
  // rather than as a finished one, and on the final screen every card is
  // installed, so the tint carried no information at all. The green moved to
  // the icon tile and the badge, where it still scans instantly.
  const stateClass = active
    ? 'border-primary/50 bg-primary/8 shadow-sm'
    : selected && !ready
      ? 'border-primary/35 bg-primary/5'
      : 'border-surface-border bg-surface-panel';

  const iconBg = ready
    ? 'bg-success/15 text-success'
    : active
      ? 'bg-primary/15 text-primary'
      : selected
        ? 'bg-primary/10 text-primary'
        : 'bg-surface-wash text-content-subtle';

  return (
    <button
      type="button"
      aria-pressed={selected}
      aria-disabled={!canToggle}
      disabled={!canToggle}
      onClick={canToggle ? onToggle : undefined}
      className={`group rounded-box border w-full text-start transition-all duration-150 ${stateClass} ${
        canToggle ? 'hover:border-primary/45 hover:bg-primary/8 cursor-pointer' : 'cursor-default'
      }`}
    >
      <div className="flex items-start gap-3 p-4">
        <div className={`setup-tile transition-colors ${iconBg}`}>
          {ready ? (
            <Check size={18} strokeWidth={2.5} />
          ) : active ? (
            <span className="loading loading-spinner loading-sm" />
          ) : (
            <Icon size={18} strokeWidth={1.6} />
          )}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium text-content-strong">{name}</p>
            {/* Once a model is installed, whether it *was* required is a
                fact about a decision already made. Showing "Required" and
                "Installed" side by side spent two badges saying one thing:
                this model is here and staying. */}
            {ready && !active ? (
              <span className="badge badge-xs badge-soft badge-success">{t`Installed`}</span>
            ) : required ? (
              <span className="badge badge-xs badge-soft gap-1 text-content-subtle">
                <Lock size={9} strokeWidth={2.2} />
                {t`Required`}
              </span>
            ) : (
              <span className="badge badge-xs badge-soft text-content-subtle">{t`Optional`}</span>
            )}
          </div>

          {active && progress ? (
            <div className="mt-2 space-y-1">
              <progress
                className="progress progress-primary h-1.5 w-full"
                value={progress.fileIndex + 1}
                max={progress.fileCount}
                aria-label={t`Download progress`}
              />
              <div className="flex items-center justify-between text-xs text-content-subtle tabular-nums">
                <span className="truncate font-mono">{progress.fileName}</span>
                <span className="ms-2 shrink-0 font-medium text-primary">{pct}%</span>
              </div>
            </div>
          ) : (
            <p className="mt-1 text-xs leading-snug text-content-subtle">{description}</p>
          )}
        </div>

        {!ready && !active && canToggle && (
          <div
            className={`mt-1 flex size-5 shrink-0 items-center justify-center rounded border-2 transition-colors ${
              selected
                ? 'border-primary bg-primary'
                : 'border-surface-border-strong group-hover:border-primary/60'
            }`}
            aria-hidden="true"
          >
            {selected && <Check size={13} strokeWidth={3} className="text-primary-content" />}
          </div>
        )}
      </div>
    </button>
  );
}
