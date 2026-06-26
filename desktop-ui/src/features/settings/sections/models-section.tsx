import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import {
  AlertTriangle,
  Check,
  Download,
  HardDrive,
  Loader2,
  Lock,
  type LucideIcon,
  Mic,
  Trash2,
  Wand2,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { workspaceStore } from '../../../store/workspace-store';
import { SettingsSection } from '../components/section';
import { type DownloadProgress, type ModelStatus, useModelStatus } from '../hooks/use-model-status';

function modelIcon(localName: string): LucideIcon {
  return localName.includes('asr') || localName.includes('whisper') ? Mic : Wand2;
}

export function ModelsSection() {
  const { t } = useLingui();
  // The model list is config-dependent: the correction LLM toggle and the ASR
  // backend/model selection (both on the Transcription tab) decide which cards
  // belong here, so we re-check status whenever any of them changes.
  const llmEnabled = useStore(workspaceStore, (s) => s.settings.llmEnabled);
  const asrBackend = useStore(workspaceStore, (s) => s.settings.asrBackend);
  const whisperModel = useStore(workspaceStore, (s) => s.settings.whisperModel);

  const { models, loading, downloading, progress, refresh, download, remove } = useModelStatus();
  const [confirmDelete, setConfirmDelete] = useState<ModelStatus | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  // These settings change which models belong in the list, so re-check on each.
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

  const modelCards = loading
    ? [0, 1].map((i) => (
        <div
          key={i}
          className="rounded-box border border-surface-border bg-surface-inset p-4 flex items-center gap-3"
        >
          <div className="skeleton size-10 rounded-field shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="skeleton h-4 w-32" />
            <div className="skeleton h-3 w-56" />
          </div>
          <div className="skeleton h-8 w-24 rounded-field" />
        </div>
      ))
    : models.map((model) => (
        <ModelManagementCard
          key={model.localName}
          model={model}
          icon={modelIcon(model.localName)}
          isDownloading={downloading === model.localName}
          anyDownloading={!!downloading}
          progress={downloading === model.localName ? progress : null}
          onDownload={() => download(model.localName)}
          onDelete={() => handleDeleteClick(model)}
        />
      ));

  return (
    <div className="flex flex-col gap-space-md">
      {/* The models themselves — download / delete the on-device ONNX files. */}
      <SettingsSection
        icon={HardDrive}
        title={t`Models`}
        description={t`AI models stored on your device.`}
      >
        {modelCards}
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
  icon: LucideIcon;
  isDownloading: boolean;
  anyDownloading: boolean;
  progress: DownloadProgress | null;
  onDownload: () => void;
  onDelete: () => void;
}

function ModelManagementCard({
  model,
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
            {model.required ? (
              <span className="badge badge-xs badge-soft gap-1 text-content-subtle">
                <Lock size={9} strokeWidth={2.2} />
                {t`Required`}
              </span>
            ) : (
              <span className="badge badge-xs badge-soft text-content-subtle">{t`Optional`}</span>
            )}
            {model.ready && !isDownloading && (
              <span className="badge badge-xs badge-soft badge-success">{t`Installed`}</span>
            )}
            {!model.ready && !isDownloading && (
              <span className="badge badge-xs badge-soft badge-error">{t`Not installed`}</span>
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
            {model.required && (
              <p className="py-2 text-sm text-error flex gap-2 items-start">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                {t`This is a required model. Dictation will not work without it.`}
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
