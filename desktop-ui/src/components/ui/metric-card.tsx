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
    <Panel tone="soft" className="metric-card">
      <div className="metric-head">
        <span className="metric-label">{label}</span>
        {icon ? <span className="metric-icon">{icon}</span> : null}
      </div>
      <div className="metric-body">
        <div className="value-display">{value}</div>
        <p className="metric-note">{note}</p>
      </div>
    </Panel>
  );
}
