import { SelectField, TextField } from "../../../components/ui/field";
import { Toggle } from "../../../components/ui/toggle";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { llmModels } from "../../setup/data/settings-data";

interface Props {
  settings: WorkspaceSettings;
}

export function CleanupSection({ settings }: Props) {
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
