import { useLingui } from '@lingui/react/macro';
import { Monitor, Moon, Palette, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';
import { activateLocale, type LocaleKey, locales } from '../../../i18n';
import {
  getThemePreference,
  isDarkTheme,
  setThemePreference,
  themeLabel,
  watchSystemTheme,
} from '../../../lib/theme';
import { saveWorkspaceSetting, type WorkspaceSettings } from '../../../store/workspace-store';
import { SettingsSection } from '../components/section';
import { ThemePicker } from '../components/theme-picker';

export function AppearanceSection({ settings }: { settings: WorkspaceSettings }) {
  const { t } = useLingui();
  const [theme, setTheme] = useState(getThemePreference);

  useEffect(() => {
    return watchSystemTheme(() => setTheme(getThemePreference()));
  }, []);

  const currentLabel = theme === 'system' ? t`System` : themeLabel(theme);
  const dark = isDarkTheme(theme) || theme === 'system';

  function selectMode(mode: 'dark' | 'light' | 'system') {
    const next = mode === 'dark' ? 'lune' : mode === 'light' ? 'lune-light' : 'system';
    setThemePreference(next);
    setTheme(next);
  }

  const activeMode = theme === 'system' ? 'system' : dark ? 'dark' : 'light';

  return (
    <SettingsSection
      title={t`Appearance`}
      description={t`Choose how Lune Voice looks.`}
      icon={Palette}
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium">{t`Mode`}</span>
          <div className="grid grid-cols-3 gap-2 sm:max-w-sm">
            {(
              [
                ['dark', Moon, t`Dark`],
                ['light', Sun, t`Light`],
                ['system', Monitor, t`System`],
              ] as const
            ).map(([mode, Icon, label]) => {
              const active = activeMode === mode;
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => selectMode(mode)}
                  className={`flex flex-col items-center gap-1 rounded-box border-2 px-3 py-2 transition-all ${
                    active
                      ? 'border-primary bg-primary/8 text-primary'
                      : 'border-surface-border bg-base-200/30 text-content-muted hover:border-base-300 hover:text-content-default'
                  }`}
                >
                  <Icon size={20} strokeWidth={1.6} />
                  <span className="text-xs font-medium">{label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium">{t`Theme`}</span>
          <div className="flex items-center justify-between rounded-field border border-surface-border bg-base-200/40 px-4 py-3">
            <div className="flex items-center gap-3">
              <div className="flex gap-1">
                <div className="size-4 rounded-full bg-primary" />
                <div className="size-4 rounded-full bg-secondary" />
                <div className="size-4 rounded-full bg-accent" />
              </div>
              <span className="text-sm text-content-muted">{currentLabel}</span>
            </div>
            <ThemePicker />
          </div>
        </div>

        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <span className="text-sm font-medium">{t`Language`}</span>
          <select
            className="select w-full sm:w-auto sm:min-w-56"
            value={settings.language}
            onChange={(event) => {
              const lang = event.target.value as LocaleKey;
              void saveWorkspaceSetting('language', lang);
              void activateLocale(lang);
            }}
          >
            {(Object.entries(locales) as [LocaleKey, string][]).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </SettingsSection>
  );
}
