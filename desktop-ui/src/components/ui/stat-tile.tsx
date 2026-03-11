import type { ReactNode } from "react";
import { cn } from "../../lib/cn";

interface StatTileProps {
  label: string;
  value: ReactNode;
  note: ReactNode;
  className?: string;
  valueClassName?: string;
  noteClassName?: string;
}

export function StatTile({
  label,
  value,
  note,
  className,
  valueClassName,
  noteClassName,
}: StatTileProps) {
  return (
    <div className={cn("stat-tile", className)}>
      <div className="stat-label">{label}</div>
      <div className={cn("stat-value mt-3", valueClassName)}>{value}</div>
      <p className={cn("stat-note mt-2", noteClassName)}>{note}</p>
    </div>
  );
}
