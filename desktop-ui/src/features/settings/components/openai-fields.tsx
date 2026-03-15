import { TextField } from "../../../components/ui/field";
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow } from "../../setup/components/settings-section";

type StringKeys<T> = { [K in keyof T]: T[K] extends string ? K : never }[keyof T];

interface Props {
  settings: WorkspaceSettings;
  baseUrlKey: StringKeys<WorkspaceSettings>;
  apiKeyKey: StringKeys<WorkspaceSettings>;
  modelKey: StringKeys<WorkspaceSettings>;
  apiDescription: string;
  modelDescription: string;
  modelPlaceholder: string;
}

export function OpenAiFields({ settings, baseUrlKey, apiKeyKey, modelKey, apiDescription, modelDescription, modelPlaceholder }: Props) {
  return (
    <>
      <SettingsRow
        label="API endpoint"
        description={`Base URL for the OpenAI-compatible ${apiDescription}.`}
        control={
          <TextField
            value={settings[baseUrlKey] as string}
            onChange={(event) => void saveWorkspaceSetting(baseUrlKey, event.target.value)}
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
            value={settings[apiKeyKey] as string}
            onChange={(event) => void saveWorkspaceSetting(apiKeyKey, event.target.value)}
            className="w-full sm:w-auto sm:min-w-64"
            placeholder="sk-…"
          />
        }
      />
      <SettingsRow
        label="Model"
        description={modelDescription}
        control={
          <TextField
            value={settings[modelKey] as string}
            onChange={(event) => void saveWorkspaceSetting(modelKey, event.target.value)}
            className="w-full sm:w-auto sm:min-w-48"
            placeholder={`${modelPlaceholder}…`}
          />
        }
      />
    </>
  );
}
