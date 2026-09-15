// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Send, Volume2 } from 'lucide-react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { UinputRuleRow } from '../components/uinput-rule-row';
import { deliveryMethods, localizeOptions, terminalModes } from '../data/settings-data';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

export function OutputSection({ settings }: SettingsProps) {
  const { t, i18n } = useLingui();
  const [cueVolume, setCueVolume] = useDebouncedSetting('soundCueVolume', settings.soundCueVolume);
  const pastes = settings.dictationMode !== 'TypeInPlace';

  return (
    <div className="flex flex-col gap-space-md">
      {/* Delivery — how finished text reaches the focused app. */}
      <SettingsSection
        title={t`Text delivery`}
        description={t`Choose how finished text is inserted into the app you are currently using.`}
        icon={Send}
      >
        <SettingsRow
          label={t`Send method`}
          description={t`Paste is simplest. Typing keystrokes can work better in some apps.`}
          control={
            <DescriptiveDropdown
              value={settings.dictationMode}
              options={localizeOptions(i18n, deliveryMethods)}
              ariaLabel={t`Send method`}
              onChange={(next) =>
                void saveWorkspaceSetting('dictationMode', next as typeof settings.dictationMode)
              }
            />
          }
        />
        {pastes && (
          <SettingsRow
            label={t`Paste compatibility`}
            description={t`Which paste shortcut to send. Automatic detects terminals; override if paste lands in the wrong place.`}
            control={
              <DescriptiveDropdown
                value={settings.terminalMode}
                options={localizeOptions(i18n, terminalModes)}
                ariaLabel={t`Paste compatibility`}
                onChange={(next) =>
                  void saveWorkspaceSetting('terminalMode', next as typeof settings.terminalMode)
                }
              />
            }
          />
        )}
        <SettingsRow
          label={t`Copy to clipboard`}
          description={t`Keep the dictated text on the clipboard after every dictation, so you can paste it again yourself.`}
          control={
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={settings.copyToClipboard}
              onChange={() =>
                void saveWorkspaceSetting('copyToClipboard', !settings.copyToClipboard)
              }
              aria-label={t`Copy to clipboard`}
            />
          }
        />
        {/* Only renders when this app installed the rule itself. */}
        <UinputRuleRow />
      </SettingsSection>

      {/* Feedback — the sounds that signal dictation state. */}
      <SettingsSection
        title={t`Sound`}
        description={t`Audio feedback played as dictation starts, finishes, or errors.`}
        icon={Volume2}
      >
        <SettingsRow
          label={t`Sound cues`}
          description={t`Play a short sound when dictation starts, finishes, or runs into an error.`}
          control={
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={settings.soundCuesEnabled}
              onChange={() =>
                void saveWorkspaceSetting('soundCuesEnabled', !settings.soundCuesEnabled)
              }
              aria-label={t`Sound cues`}
            />
          }
        />
        {settings.soundCuesEnabled && (
          <SettingsRow
            label={t`Cue volume`}
            description={t`How loud the dictation sounds play.`}
            control={
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  className="range range-primary range-sm w-28 sm:w-36"
                  min={0}
                  max={1}
                  step={0.05}
                  value={cueVolume}
                  onChange={(e) => setCueVolume(Number(e.target.value))}
                  aria-label={t`Cue volume`}
                />
                <span className="text-sm tabular-nums text-content-muted w-12 text-right">
                  {Math.round(cueVolume * 100)}%
                </span>
              </div>
            }
          />
        )}
      </SettingsSection>
    </div>
  );
}
