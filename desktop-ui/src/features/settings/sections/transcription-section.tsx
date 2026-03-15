import { SelectField } from "../../../components/ui/field";
import { Toggle } from "../../../components/ui/toggle";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { asrBackends } from "../../setup/data/settings-data";
import { OpenAiFields } from "../components/openai-fields";

interface Props {
  settings: WorkspaceSettings;
}

export function TranscriptionSection({ settings }: Props) {
  return (
    <SettingsSection title="Recognition" description="Control how your speech is turned into text.">
      <SettingsRow
        label="Voice mode"
        description="Balanced works well for most people. Compact uses less memory."
        control={
          <SelectField
            value={settings.asrBackend}
            onChange={(event) => void saveWorkspaceSetting("asrBackend", event.target.value as typeof settings.asrBackend)}
            className="w-full sm:w-auto sm:min-w-64"
          >
            {asrBackends.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>
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
        label="Use hardware acceleration"
        description="When available, this can make transcription finish faster."
        control={<Toggle checked={settings.useGpu} onChange={() => void saveWorkspaceSetting("useGpu", !settings.useGpu)} aria-label="Use hardware acceleration" />}
      />
      <SettingsRow
        label="Word-level alignment"
        description="Use forced aligner for precise word timestamps and chunk merging."
        control={
          <Toggle
            checked={settings.alignerEnabled}
            onChange={() => void saveWorkspaceSetting("alignerEnabled", !settings.alignerEnabled)}
            aria-label="Word-level alignment"
          />
        }
      />
    </SettingsSection>
  );
}
