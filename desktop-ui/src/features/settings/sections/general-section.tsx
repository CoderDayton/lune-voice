import { useLingui } from '@lingui/react/macro';
import { SlidersHorizontal } from 'lucide-react';
import { useMemo } from 'react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { activateLocale, type LocaleKey, locales } from '../../../i18n';
import { saveWorkspaceSetting, type WorkspaceSettings } from '../../../store/workspace-store';
import { SettingsRow, SettingsSection } from '../components/section';
import {
  type DescriptiveOptionData,
  type HistoryRetentionValue,
  historyRetentionOptions,
} from '../data/settings-data';

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
      icon={SlidersHorizontal}
    >
      <SettingsRow
        label={t`Language`}
        control={
          <DescriptiveDropdown
            value={settings.language}
            options={localeOptions}
            ariaLabel={t`Language`}
            onChange={(lang) => {
              void saveWorkspaceSetting('language', lang);
              void activateLocale(lang);
            }}
          />
        }
      />

      <SettingsRow
        label={t`Start minimized`}
        description={t`Launch into the system tray instead of opening the main window. Useful when Lune Voice autostarts on login.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.startMinimized}
            onChange={() => void saveWorkspaceSetting('startMinimized', !settings.startMinimized)}
            aria-label={t`Start minimized`}
          />
        }
      />

      <SettingsRow
        label={t`History retention`}
        description={t`How long finished dictations are kept on this device.`}
        control={
          <DescriptiveDropdown
            value={
              (settings.sessionHistoryDays == null
                ? 'off'
                : String(settings.sessionHistoryDays)) as HistoryRetentionValue
            }
            options={historyRetentionOptions}
            ariaLabel={t`History retention`}
            onChange={(next) =>
              void saveWorkspaceSetting('sessionHistoryDays', next === 'off' ? null : Number(next))
            }
          />
        }
      />
    </SettingsSection>
  );
}
