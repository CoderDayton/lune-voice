import { useLingui } from '@lingui/react/macro';
import { SlidersHorizontal } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
  DescriptiveDropdown,
  type DescriptiveOption,
} from '../../../components/ui/descriptive-dropdown';
import { activateLocale, type LocaleKey, locales } from '../../../i18n';
import { backendClient } from '../../../lib/backend/client';
import { saveWorkspaceSetting, type WorkspaceSettings } from '../../../store/workspace-store';
import { SettingsRow, SettingsSection } from '../components/section';
import {
  historyRetentionDays,
  historyRetentionOptions,
  historyRetentionValue,
  localizeOptions,
} from '../data/settings-data';

export function GeneralSection({ settings }: { settings: WorkspaceSettings }) {
  const { t, i18n } = useLingui();
  const [version, setVersion] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void backendClient.appVersion().then((v) => {
      if (!cancelled) setVersion(v);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // The result arrives in the app's own update modal (mounted by the shell), so
  // the button only has to report that the check started — and re-enable itself
  // so a failed check can be retried without reopening Settings.
  async function checkForUpdates() {
    setChecking(true);
    try {
      await backendClient.startUpdateCheck();
    } finally {
      setChecking(false);
    }
  }
  // Locale names stay in their own language, so these are plain labels rather
  // than messages to translate.
  const localeOptions = useMemo<DescriptiveOption<LocaleKey>[]>(
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
            value={historyRetentionValue(settings.sessionHistoryDays)}
            options={localizeOptions(i18n, historyRetentionOptions)}
            ariaLabel={t`History retention`}
            onChange={(next) =>
              void saveWorkspaceSetting('sessionHistoryDays', historyRetentionDays(next))
            }
          />
        }
      />

      <SettingsRow
        label={t`Check for updates on launch`}
        description={t`Contact the release server shortly after startup and tell you only when a newer version exists. Off by default — dictation itself never needs the network.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.checkUpdatesOnLaunch}
            onChange={() =>
              void saveWorkspaceSetting('checkUpdatesOnLaunch', !settings.checkUpdatesOnLaunch)
            }
            aria-label={t`Check for updates on launch`}
          />
        }
      />

      <SettingsRow
        label={t`Updates`}
        description={
          version
            ? t`You are running version ${version}.`
            : t`Check whether a newer version has been released.`
        }
        control={
          <button
            type="button"
            className="btn"
            disabled={checking}
            onClick={() => void checkForUpdates()}
          >
            {checking ? t`Checking…` : t`Check for updates`}
          </button>
        }
      />
    </SettingsSection>
  );
}
