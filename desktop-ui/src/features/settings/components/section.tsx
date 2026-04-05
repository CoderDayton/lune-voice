import type { ComponentType, ReactNode } from "react";
import type { WorkspaceSettings } from "../../../store/workspace-store";

export interface SettingsProps {
  settings: WorkspaceSettings;
}

interface SettingsSectionProps {
  title: string;
  description: string;
  icon?: ComponentType<{ size?: number; strokeWidth?: number; className?: string }>;
  children: ReactNode;
}

interface SettingsRowProps {
  label: string;
  description?: string;
  control: ReactNode;
}

export function SettingsSection({ title, description, icon: Icon, children }: SettingsSectionProps) {
  return (
    <section className="card border border-surface-border bg-surface-card rounded-box">
      <div className="p-5 sm:p-6 pb-0 sm:pb-0">
        <div className="flex items-center gap-3">
          {Icon && (
            <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/8">
              <Icon size={18} strokeWidth={1.8} className="text-primary/60" />
            </div>
          )}
          <div>
            <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
            <p className="text-sm text-content-subtle mt-0.5">{description}</p>
          </div>
        </div>
      </div>
      <div className="p-5 sm:p-6 flex flex-col gap-3">{children}</div>
    </section>
  );
}

export function SettingsRow({ label, description, control }: SettingsRowProps) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between py-3 border-b border-surface-border last:border-b-0">
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        {description ? <div className="text-xs text-content-subtle mt-0.5 max-w-md">{description}</div> : null}
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}
