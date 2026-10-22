// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { AlertTriangle, Check, CircleAlert, Download, Loader2, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { formatBytes } from '../../../lib/format-bytes';
import { saveWorkspaceSettings, type WorkspaceSettings } from '../../../store/workspace-store';
import {
  type AsrBackendValue,
  dictationLanguageName,
  engineAcceptsLanguage,
  formatLanguageList,
  localizeText,
  type ModelChoice,
  modelChoiceSettings,
  parakeetLanguages,
  parakeetModels,
  parakeetSupportsLanguage,
  systemLanguages,
  whisperModels,
} from '../data/settings-data';
import { type ModelStatusApi, useModelStatus } from '../hooks/use-model-status';
import { OpenAiFields } from './openai-fields';

type LocalModel = (typeof whisperModels)[number] | (typeof parakeetModels)[number];

interface DeleteTarget {
  localName: string;
  name: string;
  inUse: boolean;
}

export function AsrEngineCards({ settings }: { settings: WorkspaceSettings }) {
  const { t, i18n } = useLingui();
  const status = useModelStatus();
  const [tab, setTab] = useState<AsrBackendValue>(settings.asrBackend);
  const [deleting, setDeleting] = useState<DeleteTarget | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  function choose(choice: ModelChoice) {
    void saveWorkspaceSettings(
      modelChoiceSettings(choice, settings.dictationLanguage, [...systemLanguages(), i18n.locale]),
    );
  }

  function askDelete(target: DeleteTarget) {
    setDeleting(target);
    dialogRef.current?.showModal();
  }

  function closeDelete() {
    dialogRef.current?.close();
    setDeleting(null);
  }

  function confirmDelete() {
    if (deleting) void status.remove(deleting.localName);
    closeDelete();
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: asrBackend/whisperModel/parakeetModel are deliberate re-run triggers.
  useEffect(() => {
    status.refresh();
  }, [status.refresh, settings.asrBackend, settings.whisperModel, settings.parakeetModel]);

  const row = (m: LocalModel, inUse: boolean, choice: ModelChoice) => {
    const name = localizeText(i18n, m.label);
    return (
      <ModelRow
        key={m.value}
        label={name}
        description={m.description === undefined ? '' : localizeText(i18n, m.description)}
        size={m.size}
        localName={m.localName}
        inUse={inUse}
        status={status}
        onSelect={() => choose(choice)}
        onDelete={() => askDelete({ localName: m.localName, name, inUse })}
      />
    );
  };

  const installedBytes = status.models.reduce(
    (sum, m) => sum + (m.ready && m.sizeBytes != null ? m.sizeBytes : 0),
    0,
  );

  return (
    <div className="flex flex-col gap-4">
      <EngineTabs tab={tab} inUse={settings.asrBackend} onSelect={setTab} />
      <EngineFacts tab={tab} />

      {tab === 'WhisperCpp' && (
        <ModelList>
          {whisperModels.map((m) =>
            row(m, settings.asrBackend === 'WhisperCpp' && settings.whisperModel === m.value, {
              engine: 'WhisperCpp',
              model: m.value,
            }),
          )}
        </ModelList>
      )}

      {tab === 'ParakeetCpp' && (
        <>
          <ModelList>
            {parakeetModels.map((m) =>
              row(m, settings.asrBackend === 'ParakeetCpp' && settings.parakeetModel === m.value, {
                engine: 'ParakeetCpp',
                model: m.value,
              }),
            )}
          </ModelList>
          {settings.asrBackend === 'ParakeetCpp' && <ParakeetLanguageWarning settings={settings} />}
        </>
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
              onClick={() => choose({ engine: 'OpenAiCompatible' })}
            >
              {t`Use the remote API`}
            </button>
          )}
        </div>
      )}

      {tab !== 'OpenAiCompatible' && !status.loading && (
        <div className="flex items-center justify-between text-sm">
          <span className="text-content-muted">{t`Disk usage`}</span>
          <span className="font-medium tabular-nums text-content">
            {installedBytes > 0 ? formatBytes(installedBytes) : t`Nothing installed`}
          </span>
        </div>
      )}

      <DeleteModelDialog
        dialogRef={dialogRef}
        model={deleting}
        onCancel={closeDelete}
        onConfirm={confirmDelete}
      />
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

function ModelList({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col divide-y divide-surface-border overflow-hidden rounded-box border border-surface-border bg-surface-card">
      {children}
    </div>
  );
}

interface ModelRowProps {
  label: string;
  description: string;
  size: string;
  localName: string;
  inUse: boolean;
  status: ModelStatusApi;
  onSelect: () => void;
  onDelete: () => void;
}

function ModelRow({
  label,
  description,
  size,
  localName,
  inUse,
  status,
  onSelect,
  onDelete,
}: ModelRowProps) {
  const { t } = useLingui();
  const ready = status.byName.get(localName)?.ready ?? false;
  const downloading = status.downloading === localName;
  const progress = downloading ? status.progress : null;

  return (
    <div
      className={`flex flex-col gap-3 border-s-2 px-5 py-4 transition-colors duration-fast ${
        inUse ? 'border-s-primary bg-primary/12' : 'border-s-transparent hover:bg-surface-hover'
      }`}
    >
      <div className="flex items-center gap-4">
        <button
          type="button"
          aria-pressed={inUse}
          onClick={onSelect}
          className="flex min-w-0 flex-1 items-center gap-4 text-start"
        >
          <span
            className={`flex size-[18px] shrink-0 items-center justify-center rounded-full border-[1.5px] ${
              inUse
                ? 'border-primary bg-primary text-primary-content'
                : 'border-surface-border-strong'
            }`}
          >
            {inUse && <Check size={11} strokeWidth={3.5} />}
          </span>
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="flex flex-wrap items-baseline gap-x-2 text-[15px] font-medium text-content-strong">
              {label}
              <span className="text-[13px] font-normal tabular-nums text-content-muted">
                {size}
              </span>
              {inUse && <span className="text-xs font-medium text-primary">{t`In use`}</span>}
            </span>
            <span className="text-[13px] text-content-muted">{description}</span>
          </span>
        </button>
        <span className="flex min-w-24 shrink-0 items-center justify-end gap-1">
          {status.loading ? (
            <span className="text-xs text-content-muted">{t`Checking…`}</span>
          ) : downloading ? (
            <span className="inline-flex items-center gap-1.5 text-[13px] font-medium text-primary">
              <Loader2 size={13} className="animate-spin" />
              {t`Downloading…`}
            </span>
          ) : ready ? (
            <>
              <span className="inline-flex items-center gap-1 text-[13px] font-medium text-success">
                <Check size={13} strokeWidth={2.5} />
                {t`Installed`}
              </span>
              <button
                type="button"
                className="btn btn-ghost btn-sm btn-square text-content-muted hover:text-error"
                onClick={onDelete}
                aria-label={t`Delete`}
                title={t`Delete`}
              >
                <Trash2 size={14} />
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn btn-sm btn-ghost gap-1.5 border border-surface-border-strong text-content-strong hover:border-primary"
              onClick={() => status.download(localName)}
              disabled={!!status.downloading}
            >
              <Download size={13} />
              {t`Download`}
            </button>
          )}
        </span>
      </div>

      {downloading && (
        <div className="flex flex-col gap-1 ps-[34px]">
          <progress
            className="progress progress-primary h-1.5 w-full"
            value={progress ? progress.fileIndex + 1 : undefined}
            max={progress?.fileCount ?? 1}
            aria-label={t`Download progress`}
          />
          {progress && progress.fileCount > 0 && (
            <div className="flex items-center justify-between gap-2 text-xs tabular-nums text-content-muted">
              <span className="truncate font-mono">{progress.fileName}</span>
              <span className="shrink-0 font-medium text-primary">
                {progress.fileIndex + 1}/{progress.fileCount} ·{' '}
                {Math.round(((progress.fileIndex + 1) / progress.fileCount) * 100)}%
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface DeleteModelDialogProps {
  dialogRef: React.RefObject<HTMLDialogElement | null>;
  model: DeleteTarget | null;
  onCancel: () => void;
  onConfirm: () => void;
}

function DeleteModelDialog({ dialogRef, model, onCancel, onConfirm }: DeleteModelDialogProps) {
  const { t } = useLingui();
  return (
    <dialog ref={dialogRef} className="modal" onClose={onCancel}>
      <div className="modal-box">
        {model && (
          <>
            <h3 className="text-lg font-bold">{t`Delete ${model.name}?`}</h3>
            {model.inUse && (
              <p className="flex items-start gap-2 py-2 text-sm text-warning">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                {t`This model is in use by your current configuration. Dictation may not work until you reinstall it or pick another.`}
              </p>
            )}
            <div className="modal-action">
              <button type="button" className="btn btn-soft" onClick={onCancel}>
                {t`Cancel`}
              </button>
              <button type="button" className="btn btn-error" onClick={onConfirm}>
                {t`Delete model`}
              </button>
            </div>
          </>
        )}
      </div>
      <form method="dialog" className="modal-backdrop">
        <button type="submit">{t`Close`}</button>
      </form>
    </dialog>
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
