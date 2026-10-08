// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Monitor, Moon, Palette, Sun } from 'lucide-react';
import { useEffect } from 'react';
import { ThemePicker } from '../../../components/ui/theme-picker';
import {
  isDarkTheme,
  setThemePreference,
  themeLabel,
  useThemePreference,
  watchSystemTheme,
} from '../../../lib/theme';
import { SettingsBlock, SettingsSection } from '../components/section';

export function AppearanceSection() {
  const { t } = useLingui();
  // Shared subscription, not local state: the sidebar's More menu can
  // change the theme while this section is mounted, and a private copy would
  // keep showing the old one.
  const theme = useThemePreference();

  useEffect(() => watchSystemTheme(), []);

  const currentLabel = theme === 'system' ? t`System` : themeLabel(theme);
  const dark = isDarkTheme(theme) || theme === 'system';

  function selectMode(mode: 'dark' | 'light' | 'system') {
    const next = mode === 'dark' ? 'lune' : mode === 'light' ? 'lune-light' : 'system';
    setThemePreference(next);
  }

  const activeMode = theme === 'system' ? 'system' : dark ? 'dark' : 'light';

  return (
    <SettingsSection
      title={t`Appearance`}
      description={t`Choose how Lune Voice looks.`}
      icon={Palette}
    >
      <SettingsBlock label={t`Mode`}>
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
                className={`flex flex-col items-center gap-1 rounded-box border px-3 py-2 transition-all ${
                  active
                    ? 'border-primary/50 bg-primary/8 text-primary shadow-sm'
                    : 'border-surface-border bg-surface-inset text-content-muted hover:border-surface-border-strong hover:text-content-default'
                }`}
              >
                <Icon size={20} strokeWidth={1.6} />
                <span className="text-xs font-medium">{label}</span>
              </button>
            );
          })}
        </div>
      </SettingsBlock>

      <SettingsBlock label={t`Theme`}>
        <div className="flex items-center justify-between rounded-field border border-surface-border bg-surface-inset px-4 py-3">
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
      </SettingsBlock>
    </SettingsSection>
  );
}
