
import { activateLocale, locales, type LocaleKey } from "../../../i18n";
import { saveWorkspaceSetting } from "../../../store/workspace-store";
import { type SettingsProps, SettingsRow, SettingsSection } from "../components/section";
import { activationModes } from "../data/settings-data";
import { useLingui } from "@lingui/react/macro";

export function InputSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  async function handleLanguageChange(locale: string) {
    await activateLocale(locale);
    void saveWorkspaceSetting("language", locale);
  }

  return (
    <SettingsSection title={t`Voice input`} description={t`Choose how quickly you want to enter and leave voice typing.`}>
      <SettingsRow
        label={t`Recording mode`}
        description={t`Toggle is easiest for longer dictation. Hold is better for quick bursts.`}
        control={
          <select
            className="select w-full sm:w-auto sm:min-w-52"
            value={settings.activationMode}
            onChange={(event) => void saveWorkspaceSetting("activationMode", event.target.value as typeof settings.activationMode)}
          >
            {activationModes.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        }
      />
      <SettingsRow
        label={t`Adaptive pause detection`}
        description={t`Lune Voice learns your pause rhythm so it does not cut you off too early.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.adaptiveVad}
            onChange={() => void saveWorkspaceSetting("adaptiveVad", !settings.adaptiveVad)}
            aria-label={t`Adaptive pause detection`}
          />
        }
      />
      <SettingsRow
        label={t`Pause sensitivity`}
        description={t`Lower numbers wait longer before deciding you are done speaking.`}
        control={<span className="badge badge-outline badge-sm font-mono">{settings.speechThreshold}</span>}
      />
      <SettingsRow
        label="Language"
        description={t`Select the language for the Lune Voice interface.`}
        control={
          <select
            className="select w-full sm:w-auto sm:min-w-52"
            value={settings.language}
            onChange={(event) => void handleLanguageChange(event.target.value)}
          >
            {(Object.entries(locales) as [LocaleKey, string][]).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        }
      />
    </SettingsSection>
  );
}
