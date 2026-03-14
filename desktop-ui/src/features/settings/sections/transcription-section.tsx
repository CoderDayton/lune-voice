import { SelectField, TextField } from "../../../components/ui/field";
import { Toggle } from "../../../components/ui/toggle";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { asrBackends } from "../../setup/data/settings-data";

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
        <>
          <SettingsRow
            label="API endpoint"
            description="Base URL for the OpenAI-compatible transcription API."
            control={
              <TextField
                value={settings.openaiAsrBaseUrl}
                onChange={(event) => void saveWorkspaceSetting("openaiAsrBaseUrl", event.target.value)}
                className="w-full sm:w-auto sm:min-w-64"
                placeholder="https://api.openai.com/v1…"
              />
            }
          />
          <SettingsRow
            label="API key"
            description="Your API key. Stored locally and never shared."
            control={
              <TextField
                type="password"
                value={settings.openaiAsrApiKey}
                onChange={(event) => void saveWorkspaceSetting("openaiAsrApiKey", event.target.value)}
                className="w-full sm:w-auto sm:min-w-64"
                placeholder="sk-…"
              />
            }
          />
          <SettingsRow
            label="Model"
            description="Transcription model name, e.g. whisper-1."
            control={
              <TextField
                value={settings.openaiAsrModel}
                onChange={(event) => void saveWorkspaceSetting("openaiAsrModel", event.target.value)}
                className="w-full sm:w-auto sm:min-w-48"
                placeholder="whisper-1…"
              />
            }
          />
        </>
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
