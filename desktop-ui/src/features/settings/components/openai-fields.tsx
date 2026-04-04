
import { saveWorkspaceSetting, type WorkspaceSettings } from "../../../store/workspace-store";
import { SettingsRow } from "./section";
import { useLingui } from "@lingui/react/macro";

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
  const { t } = useLingui();
  return (
    <>
      <SettingsRow
        label={t`API endpoint`}
        description={`Base URL for the OpenAI-compatible ${apiDescription}.`}
        control={
          <input
            className="input w-full sm:w-auto sm:min-w-64"
            value={settings[baseUrlKey] as string}
            onChange={(event) => void saveWorkspaceSetting(baseUrlKey, event.target.value)}
            placeholder="https://api.openai.com/v1…"
          />
        }
      />
      <SettingsRow
        label={t`API key`}
        description={t`Your API key. Leave it blank to keep the saved key.`}
        control={
          <input
            type="password"
            autoComplete="off"
            className="input w-full sm:w-auto sm:min-w-64"
            value={settings[apiKeyKey] as string}
            onChange={(event) => void saveWorkspaceSetting(apiKeyKey, event.target.value)}
            placeholder="sk-…"
          />
        }
      />
      <SettingsRow
        label="Model"
        description={modelDescription}
        control={
          <input
            className="input w-full sm:w-auto sm:min-w-48"
            value={settings[modelKey] as string}
            onChange={(event) => void saveWorkspaceSetting(modelKey, event.target.value)}
            placeholder={`${modelPlaceholder}…`}
          />
        }
      />
    </>
  );
}
