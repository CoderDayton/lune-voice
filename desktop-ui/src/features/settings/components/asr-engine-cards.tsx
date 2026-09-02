import { useLingui } from '@lingui/react/macro';
import { ArrowRight, Check, Cpu, Download, type LucideIcon, Mic, Server } from 'lucide-react';
import { useEffect } from 'react';
import { saveWorkspaceSetting, type WorkspaceSettings } from '../../../store/workspace-store';
import { asrBackends, localizeOptions, localizeText, whisperModels } from '../data/settings-data';
import { type ModelStatusApi, useModelStatus } from '../hooks/use-model-status';

/** Per-engine icon — Whisper gets a mic, cloud a cloud, everything else a chip. */
function engineIcon(value: string): LucideIcon {
  if (value === 'WhisperCpp') return Mic;
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
 * a per-size readiness badge. The full library / download manager lives on the
 * Models tab — here we only offer an inline download for the active size.
 */
export function AsrEngineCards({ settings, onManageModels }: AsrEngineCardsProps) {
  const { i18n } = useLingui();
  const status = useModelStatus();
  const isWhisper = settings.asrBackend === 'WhisperCpp';

  // Re-check status when the engine or size changes so `inUse`/`ready` stay
  // current for the active selection.
  // biome-ignore lint/correctness/useExhaustiveDependencies: asrBackend/whisperModel are deliberate re-run triggers.
  useEffect(() => {
    status.refresh();
  }, [status.refresh, settings.asrBackend, settings.whisperModel]);

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
            onSelect={() => void saveWorkspaceSetting('asrBackend', engine.value)}
          />
        ))}
      </div>

      {isWhisper && (
        <WhisperModelPicker settings={settings} status={status} onManageModels={onManageModels} />
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
  const selectedStatus = selected ? status.byName.get(selected.localName) : undefined;
  const selectedReady = selectedStatus?.ready ?? false;
  const isDownloadingSelected = !!selected && status.downloading === selected.localName;
  const pct =
    status.progress && status.progress.fileCount > 0
      ? Math.round(((status.progress.fileIndex + 1) / status.progress.fileCount) * 100)
      : 0;

  return (
    <div className="rounded-box border border-surface-border bg-surface-inset/60 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-content-subtle">
        <Mic size={12} strokeWidth={2} />
        {t`Whisper model size`}
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        {whisperModels.map((m) => (
          <WhisperSizeCard
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
      {selected && !selectedReady && !status.loading && (
        <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-surface-border/60 pt-3">
          {isDownloadingSelected ? (
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
                onClick={() => status.download(selected.localName)}
                disabled={!!status.downloading}
              >
                <Download size={13} />
                {t`Download ${localizeText(i18n, selected.label)}`}
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
      )}
    </div>
  );
}

interface WhisperSizeCardProps {
  label: string;
  size: string;
  description: string;
  localName: string;
  active: boolean;
  status: ModelStatusApi;
  onSelect: () => void;
}

function WhisperSizeCard({
  label,
  size,
  description,
  localName,
  active,
  status,
  onSelect,
}: WhisperSizeCardProps) {
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
