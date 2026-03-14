import { SelectField, TextField } from "../../components/ui/field";
import { Toggle } from "../../components/ui/toggle";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../setup/components/settings-section";
import { asrBackends, llmModels } from "../setup/data/settings-data";

interface SectionProps {
  settings: WorkspaceSettings;
}

export function TranscriptionSection({ settings }: SectionProps) {
  return (
    <SettingsSection title="Transcription quality" description="Control how your speech is turned into text.">
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

export function CleanupSection({ settings }: SectionProps) {
  return (
    <SettingsSection title="Smart cleanup" description="Give uncertain words a second look before text is sent to your app.">
      <SettingsRow
        label="Automatic cleanup"
        description="Use a local language model to tidy up uncertain phrases."
        control={
          <Toggle
            checked={settings.llmCorrection}
            onChange={() => void saveWorkspaceSetting("llmCorrection", !settings.llmCorrection)}
            aria-label="Automatic cleanup"
          />
        }
      />
      <SettingsRow
        label="Cleanup model"
        control={
          <SelectField
            value={settings.llmBackend}
            onChange={(event) => void saveWorkspaceSetting("llmBackend", event.target.value as typeof settings.llmBackend)}
            className="w-full sm:w-auto sm:min-w-64"
          >
            {llmModels.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </SelectField>
        }
      />
      {settings.llmBackend === "OpenAiCompatible" && (
        <>
          <SettingsRow
            label="API endpoint"
            description="Base URL for the OpenAI-compatible chat completion API."
            control={
              <TextField
                value={settings.openaiLlmBaseUrl}
                onChange={(event) => void saveWorkspaceSetting("openaiLlmBaseUrl", event.target.value)}
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
                value={settings.openaiLlmApiKey}
                onChange={(event) => void saveWorkspaceSetting("openaiLlmApiKey", event.target.value)}
                className="w-full sm:w-auto sm:min-w-64"
                placeholder="sk-…"
              />
            }
          />
          <SettingsRow
            label="Model"
            description="Chat completion model name, e.g. gpt-4o-mini."
            control={
              <TextField
                value={settings.openaiLlmModel}
                onChange={(event) => void saveWorkspaceSetting("openaiLlmModel", event.target.value)}
                className="w-full sm:w-auto sm:min-w-48"
                placeholder="gpt-4o-mini…"
              />
            }
          />
        </>
      )}
      <SettingsRow
        label="Cleanup sensitivity"
        description="Words below this score are candidates for cleanup."
        control={<span className="badge badge-outline badge-sm font-mono">{settings.confidenceThreshold}</span>}
      />
    </SettingsSection>
  );
}
