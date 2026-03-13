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
    <div className={cn("stat bg-base-300/50 rounded-box border border-base-300/30 p-4", className)}>
      <div className="stat-title">{label}</div>
      <div className={cn("stat-value text-2xl", valueClassName)}>{value}</div>
      <div className={cn("stat-desc mt-1", noteClassName)}>{note}</div>
    </div>
  );
}
