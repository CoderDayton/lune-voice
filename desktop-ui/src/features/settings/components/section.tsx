// SPDX-License-Identifier: GPL-3.0-or-later

import type { ComponentType, ReactNode } from 'react';
import { PanelHeader } from '../../../components/ui/panel-header';
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
        <PanelHeader icon={Icon} title={title} description={description} />
      </div>
      <div className="p-5 sm:p-6 flex flex-col gap-3">{children}</div>
    </section>
  );
}

export function SettingsRow({ label, description, control }: SettingsRowProps) {
  return (
    <div className="settings-row hover:bg-surface-hover transition-colors duration-fast">
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

interface SettingsBlockProps {
  label: string;
  description?: string;
  children: ReactNode;
}

/**
 * Full-width settings entry: label/description stacked above a wide control
 * (cards, grids) that the right-aligned `SettingsRow` slot is too narrow for.
 */
export function SettingsBlock({ label, description, children }: SettingsBlockProps) {
  return (
    <div className="settings-row flex-col items-stretch gap-3">
      <div className="min-w-0">
        <div className="text-base font-medium">{label}</div>
        {description ? (
          <div className="text-sm text-content-subtle mt-0.5 max-w-md">{description}</div>
        ) : null}
      </div>
      {children}
    </div>
  );
}
