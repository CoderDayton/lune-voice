// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { CircleAlert } from 'lucide-react';
import { useEffect, useState } from 'react';
import { saveWorkspaceSettings, type WorkspaceSettings } from '../../../store/workspace-store';
import {
  type AsrBackendValue,
  dictationLanguageName,
  engineAcceptsLanguage,
  formatLanguageList,
  type ModelChoice,
  modelChoiceSettings,
  parakeetLanguages,
  parakeetModels,
  parakeetSupportsLanguage,
  systemLanguages,
  whisperModels,
} from '../data/settings-data';
import type { ModelStatusApi } from '../hooks/use-model-status';
import { InstalledModelPicker, type ModelOption } from './installed-model-picker';
import { OpenAiFields } from './openai-fields';

/** The installed models of the engine on `tab`, each marked if it is the one in use. */
function installedOptions(
  tab: AsrBackendValue,
  settings: WorkspaceSettings,
  status: ModelStatusApi,
): ModelOption[] {
  const ready = (localName: string) => status.byName.get(localName)?.ready ?? false;
  if (tab === 'WhisperCpp') {
    return whisperModels
      .filter((model) => ready(model.localName))
      .map((model) => ({
        model,
        choice: { engine: 'WhisperCpp', model: model.value },
        active: settings.asrBackend === tab && settings.whisperModel === model.value,
      }));
  }
  if (tab === 'ParakeetCpp') {
    return parakeetModels
      .filter((model) => ready(model.localName))
      .map((model) => ({
        model,
        choice: { engine: 'ParakeetCpp', model: model.value },
        active: settings.asrBackend === tab && settings.parakeetModel === model.value,
      }));
  }
  return [];
}

interface AsrEngineCardsProps {
  settings: WorkspaceSettings;
  status: ModelStatusApi;
  /** Opens the Models page, where downloads and switching live. */
  onOpenModels: () => void;
}

/**
 * The recognizer choice: which engine, and which of its installed models to
 * use. Downloading and deleting models happens on the Models page, which this
 * points to.
 */
export function AsrEngineCards({ settings, status, onOpenModels }: AsrEngineCardsProps) {
  const { t, i18n } = useLingui();
  const [tab, setTab] = useState<AsrBackendValue>(settings.asrBackend);

  function select(choice: ModelChoice) {
    void saveWorkspaceSettings(
      modelChoiceSettings(choice, settings.dictationLanguage, [...systemLanguages(), i18n.locale]),
    );
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: asrBackend/whisperModel/parakeetModel are deliberate re-run triggers.
  useEffect(() => {
    status.refresh();
  }, [status.refresh, settings.asrBackend, settings.whisperModel, settings.parakeetModel]);

  const options = installedOptions(tab, settings, status);

  return (
    <div className="flex flex-col gap-4">
      <EngineTabs tab={tab} inUse={settings.asrBackend} onSelect={setTab} />
      <EngineFacts tab={tab} />

      {tab !== 'OpenAiCompatible' && (
        <div className="flex flex-col gap-3">
          {status.loading ? (
            <p className="text-sm text-content-muted">{t`Checking…`}</p>
          ) : options.length === 0 ? (
            <p className="text-sm text-content-muted">{t`Nothing installed yet.`}</p>
          ) : (
            <InstalledModelPicker options={options} onSelect={select} />
          )}
          <button type="button" className="btn btn-soft btn-sm w-fit" onClick={onOpenModels}>
            {t`Manage models`}
          </button>
        </div>
      )}

      {tab === 'ParakeetCpp' && settings.asrBackend === 'ParakeetCpp' && (
        <ParakeetLanguageWarning settings={settings} />
      )}

      {tab === 'OpenAiCompatible' && (
        <div className="flex flex-col gap-3">
          <OpenAiFields
            settings={settings}
            baseUrlKey="openaiAsrBaseUrl"
            apiKeyKey="openaiAsrApiKey"
            modelKey="openaiAsrModel"
            endpointDescription={t`Base URL for the OpenAI-compatible transcription API.`}
            modelDescription={t`Transcription model name, e.g. whisper-1.`}
            modelPlaceholder="whisper-1"
          />
          {settings.asrBackend !== 'OpenAiCompatible' && (
            <button
              type="button"
              className="btn btn-primary btn-sm w-fit"
              onClick={() => select({ engine: 'OpenAiCompatible' })}
            >
              {t`Use the remote API`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

interface EngineTabsProps {
  tab: AsrBackendValue;
  inUse: AsrBackendValue;
  onSelect: (engine: AsrBackendValue) => void;
}

function EngineTabs({ tab, inUse, onSelect }: EngineTabsProps) {
  const { t } = useLingui();
  const engines: { value: AsrBackendValue; label: string }[] = [
    { value: 'WhisperCpp', label: t`Whisper` },
    { value: 'ParakeetCpp', label: t`Parakeet` },
    { value: 'OpenAiCompatible', label: t`Remote` },
  ];
  return (
    <div
      role="tablist"
      aria-label={t`Voice model`}
      className="inline-flex w-fit flex-wrap gap-1 rounded-field border border-surface-border bg-surface-inset p-1"
    >
      {engines.map((engine) => {
        const selected = engine.value === tab;
        return (
          <button
            key={engine.value}
            type="button"
            role="tab"
            aria-selected={selected}
            onClick={() => onSelect(engine.value)}
            className={`inline-flex items-center gap-2 rounded-[calc(var(--radius-field)-3px)] px-4 py-2 text-sm transition-colors duration-fast ${
              selected
                ? 'bg-surface-elevated font-medium text-content-strong shadow-sm ring-1 ring-surface-border-strong'
                : 'text-content hover:text-content-strong'
            }`}
          >
            {engine.label}
            {engine.value === inUse && (
              <span className="size-1.5 rounded-full bg-primary" title={t`In use`}>
                <span className="sr-only">{t`In use`}</span>
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function EngineFacts({ tab }: { tab: AsrBackendValue }) {
  const { t, i18n } = useLingui();
  const fact = (text: string) => <span>{text}</span>;
  const caution = (text: string) => (
    <span className="inline-flex items-center gap-1.5 text-content">
      <span className="size-1.5 rounded-full bg-info" />
      {text}
    </span>
  );

  return (
    <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1.5 text-sm text-content-muted">
      {tab === 'WhisperCpp' && (
        <>
          {fact(t`Nearly 100 languages`)}
          {fact(t`Detects the language you speak`)}
          {fact(t`Bigger is more accurate, but slower`)}
        </>
      )}
      {tab === 'ParakeetCpp' && (
        <>
          <details className="group">
            <summary className="cursor-pointer list-none text-content underline decoration-dotted underline-offset-4">
              {t`25 European languages`}
            </summary>
            <p className="mt-1.5 max-w-xl text-xs leading-relaxed">
              {formatLanguageList(i18n.locale, parakeetLanguages)}
            </p>
          </details>
          {fact(t`Fast on a CPU`)}
          {caution(t`No auto-detect: set your language below`)}
        </>
      )}
      {tab === 'OpenAiCompatible' && fact(t`Sends audio to an endpoint you set up`)}
    </div>
  );
}

function ParakeetLanguageWarning({ settings }: { settings: WorkspaceSettings }) {
  const { t, i18n } = useLingui();
  if (engineAcceptsLanguage('ParakeetCpp', settings.dictationLanguage)) return null;
  const languageName = dictationLanguageName(i18n, settings.dictationLanguage);

  return (
    <div role="alert" className="alert alert-warning alert-soft gap-2 text-sm sm:flex-row">
      <CircleAlert size={16} className="shrink-0" />
      <span>
        {parakeetSupportsLanguage(settings.dictationLanguage)
          ? t`Choose your dictation language below. Parakeet cannot tell Lune which language it heard, so until one is set every dictation is formatted as English.`
          : t`Parakeet cannot transcribe ${languageName}, so dictation will not start. Choose a language it covers below, or use Whisper.`}
      </span>
    </div>
  );
}
