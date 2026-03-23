
import { saveWorkspaceSetting } from "../../../store/workspace-store";
import { type SettingsProps, SettingsRow, SettingsSection } from "../components/section";
import { llmModels } from "../data/settings-data";
import { OpenAiFields } from "../components/openai-fields";
import { useLingui } from "@lingui/react/macro";

export function CleanupSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  return (
    <SettingsSection title={t`Smart cleanup`} description={t`Give uncertain words a second look before text is sent to your app.`}>
      <SettingsRow
        label={t`Automatic cleanup`}
        description={t`Use a local language model to tidy up uncertain phrases.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.llmCorrection}
            onChange={() => void saveWorkspaceSetting("llmCorrection", !settings.llmCorrection)}
            aria-label={t`Automatic cleanup`}
          />
        }
      />
      <SettingsRow
        label={t`Cleanup model`}
        control={
          <select
            className="select w-full sm:w-auto sm:min-w-64"
            value={settings.llmBackend}
            onChange={(event) => void saveWorkspaceSetting("llmBackend", event.target.value as typeof settings.llmBackend)}
          >
            {llmModels.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
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
