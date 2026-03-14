import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { labelForActivationMode } from "../../setup/data/settings-data";
import { KeyCaptureInput } from "../../shortcuts/key-capture-input";

interface Props {
  settings: WorkspaceSettings;
}

export function ShortcutsSection({ settings }: Props) {
  const toggleCancelConflict =
    settings.hotkeyToggle.trim().toLowerCase() ===
    settings.hotkeyCancel.trim().toLowerCase();

  return (
    <>
      <SettingsSection title="Keyboard controls" description="Use fast commands without opening menus in the middle of work.">
        <SettingsRow
          label="Start or stop recording"
          description="The main shortcut used for voice typing. Click the binding to change it."
          control={
            <KeyCaptureInput
              value={settings.hotkeyToggle}
              onChange={(combo) => void saveWorkspaceSetting("hotkeyToggle", combo)}
            />
          }
        />
        <SettingsRow
          label="Cancel current recording"
          description="Discard the current draft and return to idle. Click the binding to change it."
          control={
            <KeyCaptureInput
              value={settings.hotkeyCancel}
              onChange={(combo) => void saveWorkspaceSetting("hotkeyCancel", combo)}
            />
          }
        />
        <SettingsRow
          label="Recording style"
          description="This changes whether the shortcut toggles recording or only works while held."
          control={<span className="badge badge-outline badge-sm">{labelForActivationMode(settings.activationMode).split(" ")[0]}</span>}
        />
      </SettingsSection>

      {toggleCancelConflict && (
        <div role="alert" className="alert alert-warning alert-soft">
          <span>Toggle and cancel are bound to the same key combo. One will override the other.</span>
        </div>
      )}
    </>
  );
}
