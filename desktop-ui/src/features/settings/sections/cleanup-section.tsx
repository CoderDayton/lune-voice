import { SelectField } from "../../../components/ui/field";
import { Toggle } from "../../../components/ui/toggle";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../../setup/components/settings-section";
import { llmModels } from "../../setup/data/settings-data";
import { OpenAiFields } from "../components/openai-fields";
import { useLingui } from "@lingui/react/macro";

interface Props {
  settings: WorkspaceSettings;
}

export function CleanupSection({ settings }: Props) {
  const { t } = useLingui();
  return (
    <SettingsSection title={t`Smart cleanup`} description={t`Give uncertain words a second look before text is sent to your app.`}>
      <SettingsRow
        label={t`Automatic cleanup`}
        description={t`Use a local language model to tidy up uncertain phrases.`}
        control={
          <Toggle
            checked={settings.llmCorrection}
            onChange={() => void saveWorkspaceSetting("llmCorrection", !settings.llmCorrection)}
            aria-label={t`Automatic cleanup`}
          />
        }
      />
      <SettingsRow
        label={t`Cleanup model`}
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
        label={t`Cleanup sensitivity`}
        description={t`Words below this score are candidates for cleanup.`}
        control={<span className="badge badge-outline badge-sm font-mono">{settings.confidenceThreshold}</span>}
      />
    </SettingsSection>
  );
}
