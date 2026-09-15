// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import type { WorkspaceSettings } from '../../../store/workspace-store';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';
import { SettingsRow } from './section';

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

export function OpenAiFields({
  settings,
  baseUrlKey,
  apiKeyKey,
  modelKey,
  apiDescription,
  modelDescription,
  modelPlaceholder,
}: Props) {
  const { t } = useLingui();
  const [baseUrl, setBaseUrl] = useDebouncedSetting(baseUrlKey, settings[baseUrlKey] as string);
  const [apiKey, setApiKey] = useDebouncedSetting(apiKeyKey, settings[apiKeyKey] as string);
  const [model, setModel] = useDebouncedSetting(modelKey, settings[modelKey] as string);

  return (
    <>
      <SettingsRow
        label={t`API endpoint`}
        description={`Base URL for the OpenAI-compatible ${apiDescription}.`}
        control={
          <input
            className="input w-full sm:w-auto sm:min-w-64"
            value={baseUrl}
            onChange={(event) => setBaseUrl(event.target.value)}
            placeholder="https://api.openai.com/v1…"
          />
        }
      />
      <SettingsRow
        label={t`API key`}
        description={t`Your API key. Leave it blank to keep the saved key — or if the endpoint runs on your own machine or network and needs none.`}
        control={
          <input
            type="password"
            autoComplete="off"
            className="input w-full sm:w-auto sm:min-w-64"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
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
            value={model}
            onChange={(event) => setModel(event.target.value)}
            placeholder={`${modelPlaceholder}…`}
          />
        }
      />
    </>
  );
}
