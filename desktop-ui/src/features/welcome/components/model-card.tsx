// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, Lock, type LucideIcon } from 'lucide-react';
import { formatBytes } from '../../../lib/format-bytes';

interface ModelCardProps {
  name: string;
  description: string;
  required: boolean;
  icon: LucideIcon;
  ready: boolean;
  selected: boolean;
  canToggle: boolean;
  /** What selecting it would download; `null` when installed or unknown. */
  sizeBytes: number | null;
  onToggle: () => void;
}

export function ModelCard({
  name,
  description,
  required,
  icon: Icon,
  ready,
  selected,
  canToggle,
  sizeBytes,
  onToggle,
}: ModelCardProps) {
  const { t } = useLingui();

  // Card chrome. Only a pending selection tints the card. "Installed" does
  // not: a green card sitting in a stack of plain ones reads as a different
  // kind of object rather than as a finished one, and on the final screen
  // every card is installed, so the tint carried no information at all. The
  // green moved to the icon tile and the badge, where it still scans
  // instantly. Downloads in flight show in the setup checklist instead.
  const stateClass =
    selected && !ready
      ? 'border-primary/35 bg-primary/5'
      : 'border-surface-border bg-surface-panel';

  const iconBg = ready
    ? 'bg-success/15 text-success'
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
          {ready ? <Check size={18} strokeWidth={2.5} /> : <Icon size={18} strokeWidth={1.6} />}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium text-content-strong">{name}</p>
            {/* Once a model is installed, whether it *was* required is a
                fact about a decision already made. Showing "Required" and
                "Installed" side by side spent two badges saying one thing:
                this model is here and staying. */}
            {ready ? (
              <span className="badge badge-xs badge-soft badge-success">{t`Installed`}</span>
            ) : required ? (
              <span className="badge badge-xs badge-soft gap-1 text-content-subtle">
                <Lock size={9} strokeWidth={2.2} />
                {t`Required`}
              </span>
            ) : (
              <span className="badge badge-xs badge-soft text-content-subtle">{t`Optional`}</span>
            )}
            {sizeBytes !== null && sizeBytes > 0 && (
              <span className="badge badge-xs badge-soft tabular-nums text-content-subtle">
                {formatBytes(sizeBytes)}
              </span>
            )}
          </div>
          <p className="mt-1 text-xs leading-snug text-content-subtle">{description}</p>
        </div>

        {!ready && canToggle && (
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
