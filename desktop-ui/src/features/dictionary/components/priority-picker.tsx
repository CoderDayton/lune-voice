// SPDX-License-Identifier: GPL-3.0-or-later

import type { MessageDescriptor } from '@lingui/core';
import { msg } from '@lingui/core/macro';
import { useLingui } from '@lingui/react/macro';

interface Level {
  value: number;
  label: MessageDescriptor;
  /** Dot color — escalates neutral → info → warning to read as low/med/high. */
  dot: string;
}

const PRIORITY_LEVELS: readonly Level[] = [
  { value: 1, label: msg`Low`, dot: 'bg-content-subtle' },
  {
    value: 5,
    label: msg({ message: 'Med', comment: 'Short for medium priority' }),
    dot: 'bg-info',
  },
  { value: 9, label: msg`High`, dot: 'bg-warning' },
] as const;

function closestLevel(value: number): Level {
  if (value <= 3) return PRIORITY_LEVELS[0];
  if (value <= 7) return PRIORITY_LEVELS[1];
  return PRIORITY_LEVELS[2];
}

/** Static read-only indicator: a colored dot plus the level label. */
export function PriorityBadge({ value }: { value: number }) {
  const { i18n } = useLingui();
  const level = closestLevel(value);
  return (
    <span className="inline-flex select-none items-center gap-1.5 text-xs font-medium text-content-muted">
      <span className={`size-2 shrink-0 rounded-full ${level.dot}`} aria-hidden="true" />
      {i18n._(level.label)}
    </span>
  );
}

/**
 * Segmented control to choose Low / Med / High. Reads as one field alongside
 * the composer's text inputs: a bordered well with a tinted active segment.
 * The colored dots live on `PriorityBadge` only — inside the picker the
 * selection itself already carries the state.
 */
export function PriorityPicker({
  value,
  onChange,
  size = 'sm',
}: {
  value: number;
  onChange: (v: number) => void;
  size?: 'sm' | 'md';
}) {
  const { t, i18n } = useLingui();
  const current = closestLevel(value);

  return (
    <div
      role="toolbar"
      aria-label={t`Priority`}
      className={`flex w-full shrink-0 items-center gap-0.5 rounded-field border border-surface-border bg-surface-field p-0.5 sm:w-auto ${
        size === 'md' ? 'h-10' : 'h-8'
      }`}
    >
      {PRIORITY_LEVELS.map((level) => {
        const active = current === level;
        return (
          <button
            key={level.value}
            type="button"
            aria-pressed={active}
            className={`h-full grow rounded-md px-3.5 text-xs font-medium transition-colors sm:grow-0 ${
              active
                ? 'bg-primary/20 text-primary'
                : 'text-content-subtle hover:bg-surface-hover hover:text-content'
            }`}
            onClick={() => onChange(level.value)}
          >
            {i18n._(level.label)}
          </button>
        );
      })}
    </div>
  );
}
