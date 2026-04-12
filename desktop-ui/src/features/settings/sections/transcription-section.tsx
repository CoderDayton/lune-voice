
import { saveWorkspaceSetting } from "../../../store/workspace-store";
import { type SettingsProps, SettingsRow, SettingsSection } from "../components/section";
import { asrBackends } from "../data/settings-data";
import { OpenAiFields } from "../components/openai-fields";
import { useLingui } from "@lingui/react/macro";
import { AudioWaveform } from "lucide-react";

export function TranscriptionSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  return (
    <SettingsSection title="Recognition" description={t`Control how your speech is turned into text.`} icon={AudioWaveform}>
      <SettingsRow
        label={t`Voice model`}
        description={t`Parakeet TDT v3 runs locally with automatic language detection (25 languages).`}
        control={
          <select
            className="select w-full sm:w-auto sm:min-w-64"
            value={settings.asrBackend}
            onChange={(event) => void saveWorkspaceSetting("asrBackend", event.target.value as typeof settings.asrBackend)}
          >
            {asrBackends.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        }
      />
      {settings.asrBackend === "OpenAiCompatible" && (
        <OpenAiFields
          settings={settings}
          baseUrlKey="openaiAsrBaseUrl"
          apiKeyKey="openaiAsrApiKey"
          modelKey="openaiAsrModel"
          apiDescription="transcription API"
          modelDescription="Transcription model name, e.g. whisper-1."
          modelPlaceholder="whisper-1"
        />
      )}
      <SettingsRow
        label={t`Use hardware acceleration`}
        description={t`When available, this can make transcription finish faster.`}
        control={<input type="checkbox" className="toggle toggle-primary" checked={settings.useGpu} onChange={() => void saveWorkspaceSetting("useGpu", !settings.useGpu)} aria-label={t`Use hardware acceleration`} />}
      />
    </SettingsSection>
  );
}
