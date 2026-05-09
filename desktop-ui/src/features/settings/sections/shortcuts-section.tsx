import { Trans, useLingui } from '@lingui/react/macro';
import { Keyboard } from 'lucide-react';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { KeyCaptureInput } from '../components/key-capture-input';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { activationModes, labelFor } from '../data/settings-data';

export function ShortcutsSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const toggleCancelConflict =
    settings.hotkeyToggle.trim().toLowerCase() === settings.hotkeyCancel.trim().toLowerCase();

  return (
    <>
      <SettingsSection
        title={t`Keyboard controls`}
        description={t`Use fast commands without opening menus in the middle of work.`}
        icon={Keyboard}
      >
        <SettingsRow
          label={t`Start or stop recording`}
          description={t`The main shortcut used for voice typing. Click the binding to change it.`}
          control={
            <KeyCaptureInput
              value={settings.hotkeyToggle}
              onChange={(combo) => void saveWorkspaceSetting('hotkeyToggle', combo)}
            />
          }
        />
        <SettingsRow
          label={t`Cancel current recording`}
          description={t`Discard the current draft and return to idle. Click the binding to change it.`}
          control={
            <KeyCaptureInput
              value={settings.hotkeyCancel}
              onChange={(combo) => void saveWorkspaceSetting('hotkeyCancel', combo)}
            />
          }
        />
        <SettingsRow
          label={t`Recording style`}
          description={t`Whether the shortcut toggles recording or only works while held. Change in Voice input.`}
          control={
            <span className="badge badge-outline badge-sm">
              {labelFor(activationModes, settings.activationMode).split(' ')[0]}
            </span>
          }
        />
      </SettingsSection>

      {toggleCancelConflict && (
        <div role="alert" className="alert alert-warning alert-soft sm:flex-row">
          <span>
            <Trans>
              Toggle and cancel are bound to the same key combo. One will override the other.
            </Trans>
          </span>
        </div>
      )}
    </>
  );
}
