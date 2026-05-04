import type { ComponentType, ReactNode } from 'react';
import type { WorkspaceSettings } from '../../../store/workspace-store';

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

export function SettingsSection({
  title,
  description,
  icon: Icon,
  children,
}: SettingsSectionProps) {
  return (
    <section className="panel-surface">
      <div className="p-5 sm:p-6 pb-0 sm:pb-0">
        <div className="flex items-center gap-3">
          {Icon && (
            <div className="token-icon size-9">
              <Icon size={18} strokeWidth={1.8} />
            </div>
          )}
          <div>
            <h3 className="text-xl font-semibold tracking-tight">{title}</h3>
            <p className="text-base text-content-subtle mt-0.5">{description}</p>
          </div>
        </div>
      </div>
      <div className="p-5 sm:p-6 flex flex-col gap-3">{children}</div>
    </section>
  );
}

export function SettingsRow({ label, description, control }: SettingsRowProps) {
  return (
    <div className="settings-row">
      <div className="min-w-0">
        <div className="text-base font-medium">{label}</div>
        {description ? (
          <div className="text-sm text-content-subtle mt-0.5 max-w-md">{description}</div>
        ) : null}
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}
