
import { activateLocale, locales, type LocaleKey } from "../../../i18n";
import { saveWorkspaceSetting } from "../../../store/workspace-store";
import { type SettingsProps, SettingsRow, SettingsSection } from "../components/section";
import { activationModes } from "../data/settings-data";
import { useDebouncedSetting } from "../hooks/use-debounced-setting";
import { useLingui } from "@lingui/react/macro";
import { Mic } from "lucide-react";

export function InputSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const isParakeet = settings.asrBackend === "ParakeetTdt";
  const [threshold, setThreshold] = useDebouncedSetting("speechThreshold", settings.speechThreshold);

  async function handleLanguageChange(locale: string) {
    const previousLocale = settings.language as LocaleKey;
    const saved = await saveWorkspaceSetting("language", locale);
    if (saved) {
      await activateLocale(locale);
    } else {
      await activateLocale(previousLocale);
    }
  }

  return (
    <SettingsSection title={t`Voice input`} description={t`Choose how quickly you want to enter and leave voice typing.`} icon={Mic}>
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
        description={t`How quickly Lune decides you're done speaking.`}
        control={
          <div className="flex items-center gap-3">
            <input
              type="range"
              className="range range-primary range-xs w-28 sm:w-36"
              min={0.2}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              aria-label={t`Pause sensitivity`}
            />
            <span className="text-xs tabular-nums text-content-muted w-8 text-right">
              {Number(threshold) < 0.35 ? t`Slow` : Number(threshold) > 0.65 ? t`Fast` : t`Mid`}
            </span>
          </div>
        }
      />
      {!isParakeet && (
        <SettingsRow
          label={t`Language`}
          description={t`Select the language for voice recognition.`}
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
      )}
    </SettingsSection>
  );
}
