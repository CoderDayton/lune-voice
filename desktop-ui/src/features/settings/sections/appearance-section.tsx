// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Monitor, Moon, Palette, Sun } from 'lucide-react';
import { setThemeMode, useThemeMode } from '../../../lib/theme';
import { SettingsBlock, SettingsSection } from '../components/section';

export function AppearanceSection() {
  const { t } = useLingui();
  // Shared subscription, not local state: the sidebar's More menu can
  // change the mode while this section is mounted.
  const mode = useThemeMode();

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
          ).map(([value, Icon, label]) => {
            const active = mode === value;
            return (
              <button
                key={value}
                type="button"
                aria-pressed={active}
                onClick={() => setThemeMode(value)}
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
    </SettingsSection>
  );
}
