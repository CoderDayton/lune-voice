import { useLingui } from "@lingui/react/macro";

const PRIORITY_LEVELS = [
  { value: 1, label: "Low", badge: "badge-ghost" },
  { value: 5, label: "Med", badge: "badge-info badge-soft" },
  { value: 9, label: "High", badge: "badge-warning badge-soft" },
] as const;

function closestLevel(value: number) {
  if (value <= 3) return PRIORITY_LEVELS[0];
  if (value <= 7) return PRIORITY_LEVELS[1];
  return PRIORITY_LEVELS[2];
}

export function PriorityBadge({ value }: { value: number }) {
  const level = closestLevel(value);
  return (
    <span className={`badge badge-sm ${level.badge} select-none`}>
      {level.label}
    </span>
  );
}

export function PriorityPicker({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const { t } = useLingui();

  return (
    <div className="join join-horizontal" role="toolbar" aria-label={t`Priority`}>
      {PRIORITY_LEVELS.map((level) => {
        const active = closestLevel(value) === level;
        return (
          <button
            key={level.value}
            type="button"
            aria-pressed={active}
            className={`join-item btn btn-xs ${active ? "btn-primary" : "btn-ghost"}`}
            onClick={() => onChange(level.value)}
          >
            {level.label}
          </button>
        );
      })}
    </div>
  );
}
