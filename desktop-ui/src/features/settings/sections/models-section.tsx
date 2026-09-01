import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import {
  AlertTriangle,
  Check,
  Download,
  HardDrive,
  Loader2,
  type LucideIcon,
  Mic,
  Trash2,
  Wand2,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { formatBytes } from '../../../lib/format-bytes';
import { workspaceStore } from '../../../store/workspace-store';
import { GpuBackendCard } from '../components/gpu-backend-card';
import { SettingsSection } from '../components/section';
import { modelDownloadSizes } from '../data/settings-data';
import { type DownloadProgress, type ModelStatus, useModelStatus } from '../hooks/use-model-status';

function modelIcon(localName: string): LucideIcon {
  return localName.includes('asr') || localName.includes('whisper') ? Mic : Wand2;
}

/** Size label for a row: actual on-disk size when installed, else the download estimate. */
function sizeLabel(model: ModelStatus): string {
  if (model.ready && model.sizeBytes != null) return formatBytes(model.sizeBytes);
  return modelDownloadSizes[model.localName] ?? '';
}

export function ModelsSection() {
  const { t } = useLingui();
  // `inUse` flags depend on the ASR backend / Whisper size / correction-LLM
  // toggle (all on the Transcription tab), so re-check status when they change.
  const llmEnabled = useStore(workspaceStore, (s) => s.settings.llmEnabled);
  const asrBackend = useStore(workspaceStore, (s) => s.settings.asrBackend);
  const whisperModel = useStore(workspaceStore, (s) => s.settings.whisperModel);

  const { models, loading, downloading, progress, refresh, download, remove } = useModelStatus();
  const [confirmDelete, setConfirmDelete] = useState<ModelStatus | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: llmEnabled/asrBackend/whisperModel are deliberate re-run triggers, not used in the body.
  useEffect(() => {
    refresh();
  }, [refresh, llmEnabled, asrBackend, whisperModel]);

  function handleDeleteClick(model: ModelStatus) {
    setConfirmDelete(model);
    dialogRef.current?.showModal();
  }

  function handleDeleteConfirm() {
    if (!confirmDelete) return;
    const localName = confirmDelete.localName;
    dialogRef.current?.close();
    setConfirmDelete(null);
    void remove(localName);
  }

  const installedBytes = models.reduce(
    (sum, m) => sum + (m.ready && m.sizeBytes != null ? m.sizeBytes : 0),
    0,
  );

  // Library grouped by role, headers in pipeline order. Speech-to-text first.
  const groups: { role: string; label: string; items: ModelStatus[] }[] = [
    { role: 'speech-to-text', label: t`Speech-to-text`, items: [] },
  ];
  for (const m of models) {
    const group = groups.find((g) => g.role === m.role);
    if (group) group.items.push(m);
    else groups.push({ role: m.role, label: m.role, items: [m] });
  }

  function renderCard(model: ModelStatus) {
    return (
      <ModelManagementCard
        key={model.localName}
        model={model}
        size={sizeLabel(model)}
        icon={modelIcon(model.localName)}
        isDownloading={downloading === model.localName}
        anyDownloading={!!downloading}
        progress={downloading === model.localName ? progress : null}
        onDownload={() => download(model.localName)}
        onDelete={() => handleDeleteClick(model)}
      />
    );
  }

  return (
    <div className="flex flex-col gap-space-md">
      <SettingsSection
        icon={HardDrive}
        title={t`Models`}
        description={t`AI models stored on your device.`}
      >
        {loading ? (
          [0, 1].map((i) => (
            <div
              key={i}
              className="rounded-box border border-surface-border bg-surface-inset p-4 flex items-center gap-3"
            >
              <div className="skeleton-shimmer size-10 rounded-field shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="skeleton-shimmer h-4 w-32" />
                <div className="skeleton-shimmer h-3 w-56" />
              </div>
              <div className="skeleton-shimmer h-8 w-24 rounded-field" />
            </div>
          ))
        ) : (
          <>
            {groups
              .filter((g) => g.items.length > 0)
              .map((group) => (
                <div key={group.role} className="flex flex-col gap-2">
                  <h3 className="text-xs font-medium uppercase tracking-wider text-content-subtle">
                    {group.label}
                  </h3>
                  {group.items.map(renderCard)}
                </div>
              ))}
            {/* Acceleration is not a model, but it is the same kind of thing
                to manage: a large optional download that lands on disk and
                changes how transcription runs. Keeping it here means one
                screen owns every download the app can make. */}
            <GpuBackendCard />
            <div className="flex items-center justify-between border-t border-surface-border pt-3 text-sm">
              <span className="text-content-subtle">{t`Disk usage`}</span>
              <span className="font-medium tabular-nums text-content">
                {installedBytes > 0 ? formatBytes(installedBytes) : t`Nothing installed`}
              </span>
            </div>
          </>
        )}
      </SettingsSection>

      <DeleteConfirmModal
        dialogRef={dialogRef}
        model={confirmDelete}
        onCancel={() => {
          dialogRef.current?.close();
          setConfirmDelete(null);
        }}
        onConfirm={handleDeleteConfirm}
      />
    </div>
  );
}

interface ModelManagementCardProps {
  model: ModelStatus;
  size: string;
  icon: LucideIcon;
  isDownloading: boolean;
  anyDownloading: boolean;
  progress: DownloadProgress | null;
  onDownload: () => void;
  onDelete: () => void;
}

function ModelManagementCard({
  model,
  size,
  icon: Icon,
  isDownloading,
  anyDownloading,
  progress,
  onDownload,
  onDelete,
}: ModelManagementCardProps) {
  const { t } = useLingui();
  const pct =
    progress && progress.fileCount > 0
      ? Math.round(((progress.fileIndex + 1) / progress.fileCount) * 100)
      : 0;

  const stateClass = model.ready
    ? 'border-success/40 bg-success/5'
    : isDownloading
      ? 'border-primary/50 bg-primary/8'
      : 'border-surface-border bg-surface-inset';

  const iconBg = model.ready
    ? 'bg-success/15 text-success'
    : isDownloading
      ? 'bg-primary/15 text-primary'
      : 'bg-surface-elevated text-content-subtle';

  return (
    <div className={`rounded-box border p-4 transition-colors ${stateClass}`}>
      <div className="flex items-start gap-3 sm:gap-4">
        <div
          className={`flex size-10 shrink-0 items-center justify-center rounded-field ${iconBg}`}
        >
          {model.ready ? (
            <Check size={18} strokeWidth={2.5} />
          ) : isDownloading ? (
            <span className="loading loading-spinner loading-sm" />
          ) : (
            <Icon size={18} strokeWidth={1.6} />
          )}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="font-medium text-sm sm:text-base text-content-default truncate">
              {model.name}
            </p>
            {size && <span className="text-xs tabular-nums text-content-subtle">{size}</span>}
            {model.inUse && (
              <span className="badge badge-xs badge-soft badge-primary">{t`In use`}</span>
            )}
            {model.ready && !isDownloading && (
              <span className="badge badge-xs badge-soft badge-success">{t`Installed`}</span>
            )}
            {!model.ready && !isDownloading && (
              <span className="badge badge-xs badge-soft text-content-subtle">{t`Not installed`}</span>
            )}
          </div>

          {isDownloading && progress ? (
            <div className="mt-2 space-y-1">
              <progress
                className="progress progress-primary h-1.5 w-full"
                value={progress.fileIndex + 1}
                max={progress.fileCount}
                aria-label={t`Download progress`}
              />
              <div className="flex items-center justify-between text-xs text-content-subtle tabular-nums">
                <span className="truncate font-mono">{progress.fileName}</span>
                <span className="ml-2 shrink-0 font-medium text-primary">
                  {progress.fileIndex + 1}/{progress.fileCount} · {pct}%
                </span>
              </div>
            </div>
          ) : (
            model.description && (
              <p className="mt-1 text-xs sm:text-sm text-content-subtle leading-snug">
                {model.description}
              </p>
            )
          )}
        </div>

        <div className="shrink-0">
          {model.ready ? (
            <button
              type="button"
              className="btn btn-soft btn-error btn-sm gap-1.5"
              onClick={onDelete}
              disabled={isDownloading}
            >
              <Trash2 size={13} />
              {t`Delete`}
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary btn-sm whitespace-nowrap gap-1.5"
              onClick={onDownload}
              disabled={anyDownloading}
            >
              {isDownloading ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <Download size={13} />
              )}
              {isDownloading ? t`Downloading…` : t`Download`}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

interface DeleteConfirmModalProps {
  dialogRef: React.RefObject<HTMLDialogElement | null>;
  model: ModelStatus | null;
  onCancel: () => void;
  onConfirm: () => void;
}

function DeleteConfirmModal({ dialogRef, model, onCancel, onConfirm }: DeleteConfirmModalProps) {
  const { t } = useLingui();
  return (
    <dialog ref={dialogRef} className="modal">
      <div className="modal-box">
        {model && (
          <>
            <h3 className="font-bold text-lg">{t`Delete ${model.name}?`}</h3>
            {model.description && (
              <p className="py-2 text-sm text-content-subtle flex gap-2 items-start">
                <AlertTriangle size={14} className="mt-0.5 shrink-0 text-warning" />
                {model.description}
              </p>
            )}
            {model.inUse && (
              <p className="py-2 text-sm text-warning flex gap-2 items-start">
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
        <button
          type="button"
          className="btn btn-sm btn-circle btn-soft absolute right-2 top-2"
          onClick={onCancel}
          aria-label={t`Close`}
        >
          ✕
        </button>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button type="submit">close</button>
      </form>
    </dialog>
  );
}
