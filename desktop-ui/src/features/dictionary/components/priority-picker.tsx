import { useLingui } from '@lingui/react/macro';

interface Level {
  value: number;
  label: string;
  /** Dot color — escalates neutral → info → warning to read as low/med/high. */
  dot: string;
}

const PRIORITY_LEVELS: readonly Level[] = [
  { value: 1, label: 'Low', dot: 'bg-content-dim' },
  { value: 5, label: 'Med', dot: 'bg-info' },
  { value: 9, label: 'High', dot: 'bg-warning' },
] as const;

function closestLevel(value: number): Level {
  if (value <= 3) return PRIORITY_LEVELS[0];
  if (value <= 7) return PRIORITY_LEVELS[1];
  return PRIORITY_LEVELS[2];
}

/** Static read-only indicator: a colored dot plus the level label. */
export function PriorityBadge({ value }: { value: number }) {
  const level = closestLevel(value);
  return (
    <span className="inline-flex select-none items-center gap-1.5 text-xs font-medium text-content-muted">
      <span className={`size-2 shrink-0 rounded-full ${level.dot}`} aria-hidden="true" />
      {level.label}
    </span>
  );
}

/** Segmented control to choose Low / Med / High. */
export function PriorityPicker({
  value,
  onChange,
  size = 'xs',
}: {
  value: number;
  onChange: (v: number) => void;
  size?: 'xs' | 'sm' | 'md';
}) {
  const { t } = useLingui();
  const sizeClass = size === 'md' ? '' : `btn-${size}`;

  return (
    <div className="join join-horizontal w-full sm:w-auto" role="toolbar" aria-label={t`Priority`}>
      {PRIORITY_LEVELS.map((level) => {
        const active = closestLevel(value) === level;
        return (
          <button
            key={level.value}
            type="button"
            aria-pressed={active}
            className={`join-item btn grow gap-1.5 sm:grow-0 sm:min-w-20 ${sizeClass} ${active ? 'btn-primary' : 'btn-soft'}`}
            onClick={() => onChange(level.value)}
          >
            <span
              className={`size-2 shrink-0 rounded-full ${level.dot} ${active ? '' : 'opacity-70'}`}
              aria-hidden="true"
            />
            {level.label}
          </button>
        );
      })}
    </div>
  );
}
