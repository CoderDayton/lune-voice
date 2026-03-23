import type { ReactNode } from "react";
import type { WorkspaceSettings } from "../../../store/workspace-store";

export interface SettingsProps {
  settings: WorkspaceSettings;
}

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
    <section className="card bg-base-200 border border-base-300/30 rounded-box">
      <div className="p-5 sm:p-6 pb-0 sm:pb-0">
        <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
        <p className="text-sm opacity-50 mt-1">{description}</p>
      </div>
      <div className="p-5 sm:p-6 flex flex-col gap-3">{children}</div>
    </section>
  );
}

export function SettingsRow({ label, description, control }: SettingsRowProps) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between py-3 border-b border-base-300/15 last:border-b-0">
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        {description ? <div className="text-xs opacity-45 mt-0.5 max-w-md">{description}</div> : null}
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}
