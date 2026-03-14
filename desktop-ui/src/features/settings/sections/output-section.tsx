import { SelectField } from "../../../components/ui/field";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { deliveryMethods, stylePresets } from "../../setup/data/settings-data";

interface Props {
  settings: WorkspaceSettings;
}

export function OutputSection({ settings }: Props) {
  return (
    <SettingsSection title="Text output" description="Choose how finished text is inserted into the app you are currently using.">
      <SettingsRow
        label="Send method"
        description="Paste is simplest. Typing keystrokes can work better in some apps."
        control={
          <SelectField
            value={settings.dictationMode}
            onChange={(event) => void saveWorkspaceSetting("dictationMode", event.target.value as typeof settings.dictationMode)}
            className="w-full sm:w-auto sm:min-w-56"
          >
            {deliveryMethods.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>
        }
      />
      <SettingsRow
        label="Writing style"
        description="Pick the tone used when final cleanup is applied."
        control={
          <SelectField
            value={settings.stylePreset}
            onChange={(event) => void saveWorkspaceSetting("stylePreset", event.target.value)}
            className="w-full sm:w-auto sm:min-w-52"
          >
            {stylePresets.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>
        }
      />
    </SettingsSection>
  );
}
