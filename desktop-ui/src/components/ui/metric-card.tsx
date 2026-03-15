import type { ReactNode } from "react";
import { Panel } from "./panel";

interface MetricCardProps {
  label: string;
  value: string;
  note: string;
  icon?: ReactNode;
}

export function MetricCard({ label, value, note, icon }: MetricCardProps) {
  return (
    <Panel tone="soft" className="p-3 sm:p-4">
      <div className="flex items-center justify-between">
        <span className="label-text-custom">{label}</span>
        {icon ? <span className="opacity-50">{icon}</span> : null}
      </div>
      <div className="mt-3 stack-copy-tight">
        <div className="value-display">{value}</div>
        <p className="body-copy-soft max-w-full sm:max-w-48">{note}</p>
      </div>
    </Panel>
  );
}
