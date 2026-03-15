import { SelectField } from "../../../components/ui/field";
import { Toggle } from "../../../components/ui/toggle";
import { activateLocale, locales, type LocaleKey } from "../../../i18n";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { activationModes } from "../../setup/data/settings-data";

interface Props {
  settings: WorkspaceSettings;
}

export function InputSection({ settings }: Props) {
  async function handleLanguageChange(locale: string) {
    await activateLocale(locale);
    void saveWorkspaceSetting("language", locale);
  }

  return (
    <SettingsSection title="Voice input" description="Choose how quickly you want to enter and leave voice typing.">
      <SettingsRow
        label="Recording mode"
        description="Toggle is easiest for longer dictation. Hold is better for quick bursts."
        control={
          <SelectField
            value={settings.activationMode}
            onChange={(event) => void saveWorkspaceSetting("activationMode", event.target.value as typeof settings.activationMode)}
            className="w-full sm:w-auto sm:min-w-52"
          >
            {activationModes.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>
        }
      />
      <SettingsRow
        label="Adaptive pause detection"
        description="Lune Voice learns your pause rhythm so it does not cut you off too early."
        control={
          <Toggle
            checked={settings.adaptiveVad}
            onChange={() => void saveWorkspaceSetting("adaptiveVad", !settings.adaptiveVad)}
            aria-label="Adaptive pause detection"
          />
        }
      />
      <SettingsRow
        label="Pause sensitivity"
        description="Lower numbers wait longer before deciding you are done speaking."
        control={<span className="badge badge-outline badge-sm font-mono">{settings.speechThreshold}</span>}
      />
      <SettingsRow
        label="Language"
        description="Select the language for the Lune Voice interface."
        control={
          <SelectField
            value={settings.language}
            onChange={(event) => void handleLanguageChange(event.target.value)}
            className="w-full sm:w-auto sm:min-w-52"
          >
            {(Object.entries(locales) as [LocaleKey, string][]).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </SelectField>
        }
      />
    </SettingsSection>
  );
}
