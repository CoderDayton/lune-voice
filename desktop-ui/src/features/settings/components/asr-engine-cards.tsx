import { useLingui } from '@lingui/react/macro';
import { Check, Cloud, Cpu, Download, type LucideIcon, Mic } from 'lucide-react';
import { useEffect } from 'react';
import { saveWorkspaceSetting, type WorkspaceSettings } from '../../../store/workspace-store';
import { asrBackends, whisperModels } from '../data/settings-data';
import { type ModelStatusApi, useModelStatus } from '../hooks/use-model-status';

/** Per-engine icon — Whisper gets a mic, cloud a cloud, everything else a chip. */
function engineIcon(value: string): LucideIcon {
  if (value === 'WhisperCpp') return Mic;
  if (value === 'OpenAiCompatible') return Cloud;
  return Cpu;
}

/**
 * Card-based ASR engine picker for the Transcription tab. Selecting an engine
 * persists it immediately; choosing Whisper reveals model-size cards with
 * inline download / installed status for the selected size.
 */
export function AsrEngineCards({ settings }: { settings: WorkspaceSettings }) {
  const status = useModelStatus();
  const isWhisper = settings.asrBackend === 'WhisperCpp';

  // Status is reported only for the *selected* Whisper size, so re-check when
  // the engine or size changes to keep the badge/button current.
  // biome-ignore lint/correctness/useExhaustiveDependencies: asrBackend/whisperModel are deliberate re-run triggers.
  useEffect(() => {
    status.refresh();
  }, [status.refresh, settings.asrBackend, settings.whisperModel]);

  return (
    <div className="flex flex-col gap-3">
      <div className="grid gap-2 sm:grid-cols-3">
        {asrBackends.map((engine) => (
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

      {isWhisper && <WhisperModelPicker settings={settings} status={status} />}
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
}

function WhisperModelPicker({ settings, status }: WhisperPickerProps) {
  const { t } = useLingui();
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
            label={m.label}
            size={m.size}
            description={m.description ?? ''}
            localName={m.localName}
            active={settings.whisperModel === m.value}
            status={status}
            onSelect={() => void saveWorkspaceSetting('whisperModel', m.value)}
          />
        ))}
      </div>
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
  // Status is only reported for the selected size (see useModelStatus).
  const st = active ? status.byName.get(localName) : undefined;
  const isDownloading = status.downloading === localName;
  const pct =
    status.progress && status.progress.fileCount > 0
      ? Math.round(((status.progress.fileIndex + 1) / status.progress.fileCount) * 100)
      : 0;

  return (
    <div
      className={`flex flex-col rounded-field border p-2.5 transition-colors ${
        active ? 'border-primary/50 bg-primary/8' : 'border-surface-border bg-surface-elevated'
      }`}
    >
      <button type="button" aria-pressed={active} onClick={onSelect} className="text-left">
        <span className="flex items-center justify-between gap-2">
          <span
            className={`text-sm font-medium ${active ? 'text-content-strong' : 'text-content'}`}
          >
            {label}
          </span>
          {active && <Check size={14} strokeWidth={2.5} className="shrink-0 text-primary" />}
        </span>
        <span className="mt-0.5 block text-xs text-content-subtle">
          {size} · {description}
        </span>
      </button>

      {active && (
        <div className="mt-2 border-t border-surface-border/60 pt-2">
          {isDownloading ? (
            <div className="space-y-1">
              <progress
                className="progress progress-primary h-1 w-full"
                value={(status.progress?.fileIndex ?? 0) + 1}
                max={status.progress?.fileCount ?? 1}
                aria-label={t`Download progress`}
              />
              <span className="text-xs font-medium tabular-nums text-primary">
                {t`Downloading…`} {pct}%
              </span>
            </div>
          ) : status.loading || st === undefined ? (
            <span className="text-xs text-content-subtle">{t`Checking…`}</span>
          ) : st.ready ? (
            <span className="inline-flex items-center gap-1 text-xs font-medium text-success">
              <Check size={12} strokeWidth={2.5} />
              {t`Installed`}
            </span>
          ) : (
            <button
              type="button"
              className="btn btn-primary btn-xs gap-1"
              onClick={() => status.download(localName)}
              disabled={!!status.downloading}
            >
              <Download size={11} />
              {t`Download`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
