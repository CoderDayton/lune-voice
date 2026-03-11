import type { ReactNode } from "react";
import { Panel } from "../../../components/ui/panel";

interface SettingsSectionProps {
  title: string;
  description: string;
  children: ReactNode;
}

interface SettingsRowProps {
  label: string;
  description?: string;
  control: ReactNode;
}

export function SettingsSection({ title, description, children }: SettingsSectionProps) {
  return (
    <Panel className="panel-pad-lg">
      <div className="stack-copy mb-5">
        <h3 className="card-title">{title}</h3>
        <p className="body-copy-soft">{description}</p>
      </div>
      <div className="stack-section">{children}</div>
    </Panel>
  );
}

export function SettingsRow({ label, description, control }: SettingsRowProps) {
  return (
    <div className="detail-row sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <div className="supporting-label text-copy-50">{label}</div>
        {description ? <div className="body-copy-soft mt-1">{description}</div> : null}
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}
