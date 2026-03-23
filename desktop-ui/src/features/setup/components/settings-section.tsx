import type { ReactNode } from "react";

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
    <fieldset className="fieldset bg-base-200 border-base-300 rounded-box border p-4 sm:p-6">
      <legend className="fieldset-legend">{title}</legend>
      <p className="text-sm opacity-50 mb-4">{description}</p>
      <div className="flex flex-col gap-4">{children}</div>
    </fieldset>
  );
}

export function SettingsRow({ label, description, control }: SettingsRowProps) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between py-2 border-b border-base-300/20 last:border-b-0">
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        {description ? <div className="text-xs opacity-50 mt-0.5">{description}</div> : null}
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}
