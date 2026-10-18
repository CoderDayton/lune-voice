// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import {
  ArrowRight,
  Check,
  CircleAlert,
  Cpu,
  Download,
  Gauge,
  Info,
  type LucideIcon,
  Mic,
  Server,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import {
  saveWorkspaceSetting,
  saveWorkspaceSettings,
  type WorkspaceSettings,
} from '../../../store/workspace-store';
import {
  type AsrBackendValue,
  asrBackends,
  dictationLanguageName,
  engineAcceptsLanguage,
  formatLanguageList,
  languageForEngine,
  localizeOptions,
  localizeText,
  parakeetLanguages,
  parakeetModels,
  parakeetSupportsLanguage,
  systemLanguages,
  whisperModels,
} from '../data/settings-data';
import { type ModelStatusApi, useModelStatus } from '../hooks/use-model-status';

/** Per-engine icon — Whisper a mic, Parakeet a gauge (its case is speed), a
 *  server for the remote API, a chip for anything else. */
function engineIcon(value: string): LucideIcon {
  if (value === 'WhisperCpp') return Mic;
  if (value === 'ParakeetCpp') return Gauge;
  // A server, not a cloud: the endpoint may well be on localhost.
  if (value === 'OpenAiCompatible') return Server;
  return Cpu;
}

interface AsrEngineCardsProps {
  settings: WorkspaceSettings;
  /** Navigate to the Models tab (for the "Manage in Models" shortcut). */
  onManageModels?: () => void;
}

/**
 * Card-based ASR engine picker for the Transcription tab. Selecting an engine
 * persists it immediately; choosing Whisper reveals a model-size selector with
 * a per-size readiness badge, and choosing Parakeet its one model with the
 * languages it covers. The full library / download manager lives on the
 * Models tab — here we only offer an inline download for the active model.
 */
export function AsrEngineCards({ settings, onManageModels }: AsrEngineCardsProps) {
  const { i18n } = useLingui();
  const status = useModelStatus();
  const isWhisper = settings.asrBackend === 'WhisperCpp';
  const isParakeet = settings.asrBackend === 'ParakeetCpp';
  // The language an engine switch had to pick, so the Parakeet panel can say
  // so. Cleared on the next switch; never persisted.
  const [pickedLanguage, setPickedLanguage] = useState<string | null>(null);

  // Switch engine and, when the new one cannot use the current dictation
  // language, move to one it can — in the same save, so the backend never
  // sees (and refuses) the in-between combination.
  function selectEngine(backend: AsrBackendValue) {
    if (backend === settings.asrBackend) return;
    const language = languageForEngine(backend, settings.dictationLanguage, [
      ...systemLanguages(),
      i18n.locale,
    ]);
    setPickedLanguage(language === settings.dictationLanguage ? null : language);
    void saveWorkspaceSettings({ asrBackend: backend, dictationLanguage: language });
  }

  // Re-check status when the engine or size changes so `inUse`/`ready` stay
  // current for the active selection.
  // biome-ignore lint/correctness/useExhaustiveDependencies: asrBackend/whisperModel/parakeetModel are deliberate re-run triggers.
  useEffect(() => {
    status.refresh();
  }, [status.refresh, settings.asrBackend, settings.whisperModel, settings.parakeetModel]);

  return (
    <div className="flex flex-col gap-3">
      <div className="grid gap-2 sm:grid-cols-3">
        {localizeOptions(i18n, asrBackends).map((engine) => (
          <EngineCard
            key={engine.value}
            label={engine.label}
            description={engine.description ?? ''}
            icon={engineIcon(engine.value)}
            active={settings.asrBackend === engine.value}
            onSelect={() => selectEngine(engine.value)}
          />
        ))}
      </div>

      {isWhisper && (
        <WhisperModelPicker settings={settings} status={status} onManageModels={onManageModels} />
      )}
      {isParakeet && (
        <ParakeetModelPanel
          settings={settings}
          status={status}
          pickedLanguage={pickedLanguage}
          onManageModels={onManageModels}
        />
      )}
    </div>
  );
}

interface EngineCardProps {
  label: string;
  description: string;
  icon: LucideIcon;
  active: boolean;
  onSelect: () => void;
}

function EngineCard({ label, description, icon: Icon, active, onSelect }: EngineCardProps) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onSelect}
      className={`group flex flex-col gap-2 rounded-box border p-3 text-left transition-all duration-fast ${
        active
          ? 'border-primary/50 bg-primary/8 shadow-sm'
          : 'border-surface-border bg-surface-inset hover:border-primary/40 hover:bg-primary/5'
      }`}
    >
      <div className="flex items-center justify-between">
        <span
          className={`flex size-9 items-center justify-center rounded-field transition-colors ${
            active
              ? 'bg-primary/15 text-primary'
              : 'bg-surface-elevated text-content-subtle group-hover:text-content'
          }`}
        >
          <Icon size={17} strokeWidth={1.7} />
        </span>
        {active && <Check size={16} strokeWidth={2.5} className="text-primary" />}
      </div>
      <span className={`text-sm font-medium ${active ? 'text-content-strong' : 'text-content'}`}>
        {label}
      </span>
      <span className="text-xs leading-snug text-content-subtle">{description}</span>
    </button>
  );
}

interface WhisperPickerProps {
  settings: WorkspaceSettings;
  status: ModelStatusApi;
  onManageModels?: () => void;
}

function WhisperModelPicker({ settings, status, onManageModels }: WhisperPickerProps) {
  const { t, i18n } = useLingui();
  const selected = whisperModels.find((m) => m.value === settings.whisperModel);

  return (
    <div className="rounded-box border border-surface-border bg-surface-inset/60 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-content-subtle">
        <Mic size={12} strokeWidth={2} />
        {t`Whisper model size`}
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        {whisperModels.map((m) => (
          <ModelSizeCard
            key={m.value}
            label={localizeText(i18n, m.label)}
            size={m.size}
            description={m.description === undefined ? '' : localizeText(i18n, m.description)}
            localName={m.localName}
            active={settings.whisperModel === m.value}
            status={status}
            onSelect={() => void saveWorkspaceSetting('whisperModel', m.value)}
          />
        ))}
      </div>

      {/* The active size isn't installed: offer one inline download + a link to
          the full manager. Everything else is managed on the Models tab. */}
      {selected && (
        <InlineModelDownload
          localName={selected.localName}
          label={localizeText(i18n, selected.label)}
          status={status}
          onManageModels={onManageModels}
        />
      )}
    </div>
  );
}

interface ModelSizeCardProps {
  label: string;
  size: string;
  description: string;
  localName: string;
  active: boolean;
  status: ModelStatusApi;
  onSelect: () => void;
}

function ModelSizeCard({
  label,
  size,
  description,
  localName,
  active,
  status,
  onSelect,
}: ModelSizeCardProps) {
  const { t } = useLingui();
  const st = status.byName.get(localName);
  const ready = st?.ready ?? false;

  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onSelect}
      className={`flex flex-col rounded-field border p-2.5 text-left transition-colors ${
        active ? 'border-primary/50 bg-primary/8' : 'border-surface-border bg-surface-elevated'
      }`}
    >
      <span className="flex items-center justify-between gap-2">
        <span className={`text-sm font-medium ${active ? 'text-content-strong' : 'text-content'}`}>
          {label}
        </span>
        {active && <Check size={14} strokeWidth={2.5} className="shrink-0 text-primary" />}
      </span>
      <span className="mt-1">
        {status.loading ? (
          <span className="text-xs text-content-subtle">{t`Checking…`}</span>
        ) : ready ? (
          <span className="inline-flex items-center gap-1 text-xs font-medium text-success">
            <Check size={12} strokeWidth={2.5} />
            {t`Installed`}
          </span>
        ) : (
          <span className="text-xs tabular-nums text-content-subtle">{size}</span>
        )}
      </span>
      <span className="mt-0.5 block text-xs text-content-subtle">{description}</span>
    </button>
  );
}

interface InlineModelDownloadProps {
  localName: string;
  /** Model name for the button, already localized. */
  label: string;
  status: ModelStatusApi;
  onManageModels?: () => void;
}

/**
 * One inline download for the active engine's model when it isn't installed,
 * plus a shortcut to the full manager. Renders nothing once the model is ready
 * (or while readiness is still being checked).
 */
function InlineModelDownload({
  localName,
  label,
  status,
  onManageModels,
}: InlineModelDownloadProps) {
  const { t } = useLingui();
  const ready = status.byName.get(localName)?.ready ?? false;
  if (ready || status.loading) return null;
  const isDownloading = status.downloading === localName;
  const pct =
    status.progress && status.progress.fileCount > 0
      ? Math.round(((status.progress.fileIndex + 1) / status.progress.fileCount) * 100)
      : 0;

  return (
    <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-surface-border/60 pt-3">
      {isDownloading ? (
        <div className="flex-1 min-w-40 space-y-1">
          <progress
            className="progress progress-primary h-1.5 w-full"
            value={(status.progress?.fileIndex ?? 0) + 1}
            max={status.progress?.fileCount ?? 1}
            aria-label={t`Download progress`}
          />
          <span className="text-xs font-medium tabular-nums text-primary">
            {t`Downloading…`} {pct}%
          </span>
        </div>
      ) : (
        <>
          <button
            type="button"
            className="btn btn-primary btn-sm gap-1.5"
            onClick={() => status.download(localName)}
            disabled={!!status.downloading}
          >
            <Download size={13} />
            {t`Download ${label}`}
          </button>
          {onManageModels && (
            <button
              type="button"
              className="btn btn-link btn-sm gap-1 px-0 text-content-subtle no-underline hover:text-primary"
              onClick={onManageModels}
            >
              {t`Manage in Models`}
              <ArrowRight size={13} />
            </button>
          )}
        </>
      )}
    </div>
  );
}

interface ParakeetPanelProps {
  settings: WorkspaceSettings;
  status: ModelStatusApi;
  /** The dictation language just picked for Parakeet by an engine switch. */
  pickedLanguage: string | null;
  onManageModels?: () => void;
}

/**
 * The Parakeet picker: its two quantizations, the languages it covers, and
 * why it needs a dictation language set. Parakeet recognizes its languages by
 * itself but never reports which one it heard, so with Auto the rest of the
 * pipeline would format everything by English rules — the language selector
 * therefore offers no Auto with Parakeet, and this panel says why. When an
 * engine switch had to choose a language, it says which.
 */
function ParakeetModelPanel({
  settings,
  status,
  pickedLanguage,
  onManageModels,
}: ParakeetPanelProps) {
  const { t, i18n } = useLingui();
  const selected = parakeetModels.find((m) => m.value === settings.parakeetModel);
  const languages = formatLanguageList(i18n.locale, parakeetLanguages);
  const languageName = dictationLanguageName(i18n, settings.dictationLanguage);
  const languageUsable = engineAcceptsLanguage('ParakeetCpp', settings.dictationLanguage);
  const languageCovered = parakeetSupportsLanguage(settings.dictationLanguage);

  return (
    <div className="rounded-box border border-surface-border bg-surface-inset/60 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-content-subtle">
        <Gauge size={12} strokeWidth={2} />
        {t`Parakeet model size`}
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {parakeetModels.map((m) => (
          <ModelSizeCard
            key={m.value}
            label={localizeText(i18n, m.label)}
            size={m.size}
            description={m.description === undefined ? '' : localizeText(i18n, m.description)}
            localName={m.localName}
            active={settings.parakeetModel === m.value}
            status={status}
            onSelect={() => void saveWorkspaceSetting('parakeetModel', m.value)}
          />
        ))}
      </div>
      <p className="mt-2 text-xs leading-snug text-content-subtle">{t`Languages: ${languages}`}</p>

      {languageUsable ? (
        <div role="status" className="alert alert-info alert-soft mt-3 gap-2 text-sm sm:flex-row">
          <Info size={16} className="shrink-0" />
          <span>
            {pickedLanguage === settings.dictationLanguage
              ? t`Dictation language set to ${languageName}. Parakeet cannot tell Lune which language it heard, so it has no Auto-detect — change the language below whenever you dictate in another one.`
              : t`Parakeet cannot tell Lune which language it heard, so it has no Auto-detect. Set the dictation language below to the one you speak, and change it when you switch languages.`}
          </span>
        </div>
      ) : (
        // Reachable only from a hand-edited settings file: the selector and
        // the backend both refuse this combination.
        <div role="alert" className="alert alert-warning alert-soft mt-3 gap-2 text-sm sm:flex-row">
          <CircleAlert size={16} className="shrink-0" />
          <span>
            {languageCovered
              ? t`Choose your dictation language below. Parakeet cannot tell Lune which language it heard, so until one is set every dictation is formatted as English.`
              : t`Parakeet cannot transcribe ${languageName}, so dictation will not start. Choose a language it covers below, or use Whisper.`}
          </span>
        </div>
      )}

      {selected && (
        <InlineModelDownload
          localName={selected.localName}
          label={`Parakeet ${localizeText(i18n, selected.label)}`}
          status={status}
          onManageModels={onManageModels}
        />
      )}
    </div>
  );
}
