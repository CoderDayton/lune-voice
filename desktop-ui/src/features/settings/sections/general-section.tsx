import { useLingui } from '@lingui/react/macro';
import { Settings2 } from 'lucide-react';
import { useMemo } from 'react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { activateLocale, type LocaleKey, locales } from '../../../i18n';
import { saveWorkspaceSetting, type WorkspaceSettings } from '../../../store/workspace-store';
import { SettingsSection } from '../components/section';
import type { DescriptiveOptionData } from '../data/settings-data';

export function GeneralSection({ settings }: { settings: WorkspaceSettings }) {
  const { t } = useLingui();
  const localeOptions = useMemo<DescriptiveOptionData<LocaleKey>[]>(
    () =>
      (Object.entries(locales) as [LocaleKey, string][]).map(([key, label]) => ({
        value: key,
        label,
      })),
    [],
  );

  return (
    <SettingsSection
      title={t`General`}
      description={t`Application behavior and startup preferences.`}
      icon={Settings2}
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <span className="text-sm font-medium">{t`Language`}</span>
          <DescriptiveDropdown
            value={settings.language}
            options={localeOptions}
            ariaLabel={t`Language`}
            onChange={(lang) => {
              void saveWorkspaceSetting('language', lang);
              void activateLocale(lang);
            }}
          />
        </div>

        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-col">
            <span className="text-sm font-medium">{t`Start minimized`}</span>
            <span className="text-xs text-content-muted">
              {t`Launch into the system tray instead of opening the main window. Useful when Lune autostarts on login.`}
            </span>
          </div>
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.startMinimized}
            onChange={() => void saveWorkspaceSetting('startMinimized', !settings.startMinimized)}
            aria-label={t`Start minimized`}
          />
        </div>
      </div>
    </SettingsSection>
  );
}
