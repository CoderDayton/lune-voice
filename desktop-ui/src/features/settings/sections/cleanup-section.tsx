import { SelectField } from "../../../components/ui/field";
import { Toggle } from "../../../components/ui/toggle";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { llmModels } from "../../setup/data/settings-data";
import { OpenAiFields } from "../components/openai-fields";

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
        <OpenAiFields
          settings={settings}
          baseUrlKey="openaiLlmBaseUrl"
          apiKeyKey="openaiLlmApiKey"
          modelKey="openaiLlmModel"
          apiDescription="chat completion API"
          modelDescription="Chat completion model name, e.g. gpt-4o-mini."
          modelPlaceholder="gpt-4o-mini"
        />
      )}
      <SettingsRow
        label="Cleanup sensitivity"
        description="Words below this score are candidates for cleanup."
        control={<span className="badge badge-outline badge-sm font-mono">{settings.confidenceThreshold}</span>}
      />
    </SettingsSection>
  );
}
