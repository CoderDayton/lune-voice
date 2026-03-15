import { SelectField } from "../../../components/ui/field";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { deliveryMethods, stylePresets } from "../../setup/data/settings-data";
import { useLingui } from "@lingui/react/macro";

interface Props {
  settings: WorkspaceSettings;
}

export function OutputSection({ settings }: Props) {
  const { t } = useLingui();
  return (
    <SettingsSection title={t`Text output`} description={t`Choose how finished text is inserted into the app you are currently using.`}>
      <SettingsRow
        label={t`Send method`}
        description={t`Paste is simplest. Typing keystrokes can work better in some apps.`}
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
        label={t`Writing style`}
        description={t`Pick the tone used when final cleanup is applied.`}
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
