import { useLingui } from '@lingui/react/macro';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  AlertTriangle,
  Circle,
  Download,
  HardDrive,
  Info,
  Loader2,
  Trash2,
  Zap,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { SettingsSection } from '../components/section';

interface ModelStatus {
  name: string;
  repoId: string;
  localName: string;
  ready: boolean;
  description: string;
  required: boolean;
}

interface DownloadProgress {
  modelName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  done: boolean;
}

function getModelState(model: ModelStatus, ready: boolean): 'active' | 'downloaded' | 'missing' {
  if (!ready) return 'missing';
  if (model.required) return 'active';
  return 'downloaded';
}

function StatusBadge({ state }: { state: 'active' | 'downloaded' | 'missing' }) {
  switch (state) {
    case 'active':
      return (
        <span className="badge badge-success badge-sm gap-1">
          <Zap size={10} />
          Active
        </span>
      );
    case 'downloaded':
      return (
        <span className="badge badge-ghost badge-sm gap-1">
          <Circle size={10} />
          Idle
        </span>
      );
    case 'missing':
      return (
        <span className="badge badge-error badge-sm gap-1">
          <Circle size={10} />
          Not installed
        </span>
      );
  }
}

export function ModelsSection() {
  const { t } = useLingui();

  const [models, setModels] = useState<ModelStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<ModelStatus | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  function refresh() {
    invoke<ModelStatus[]>('check_model_status')
      .then(setModels)
      .catch((e) => console.error('check_model_status:', e))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    const unlisten = listen<DownloadProgress>('lune://download-progress', (event) => {
      setProgress(event.payload);
      if (event.payload.done) {
        setDownloading(null);
        setProgress(null);
        refresh();
      }
    });
    return () => {
      void unlisten.then((f) => f());
    };
  }, []);

  function handleDownload(localName: string) {
    setDownloading(localName);
    invoke('download_model_cmd', { localName }).catch((e) => {
      console.error('download_model_cmd:', e);
      setDownloading(null);
      setProgress(null);
    });
  }

  function handleDeleteClick(model: ModelStatus) {
    setConfirmDelete(model);
    dialogRef.current?.showModal();
  }

  function handleDeleteConfirm() {
    if (!confirmDelete) return;
    const localName = confirmDelete.localName;
    dialogRef.current?.close();
    setConfirmDelete(null);
    invoke('delete_model_cmd', { localName })
      .then(() => refresh())
      .catch((e) => console.error('delete_model_cmd:', e));
  }

  if (loading) {
    return (
      <SettingsSection
        title={t`Models`}
        description={t`Manage AI models stored on your device.`}
        icon={HardDrive}
      >
        {[0, 1].map((i) => (
          <div key={i} className="flex items-center gap-3 py-3">
            <div className="skeleton size-8 sm:size-10 rounded-field shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="skeleton h-4 w-24 sm:w-32" />
              <div className="skeleton h-3 w-40 sm:w-56" />
            </div>
            <div className="skeleton h-8 w-20 sm:w-24 rounded-field" />
          </div>
        ))}
      </SettingsSection>
    );
  }

  return (
    <>
      <SettingsSection
        icon={HardDrive}
        title={t`Models`}
        description={t`AI models stored on your device.`}
      >
        {models.map((model) => {
          const isDownloading = downloading === model.localName;
          const state = getModelState(model, model.ready);
          const progressText =
            progress && isDownloading
              ? `${progress.fileName} (${progress.fileIndex + 1}/${progress.fileCount})`
              : null;

          return (
            <div
              key={model.localName}
              className="flex flex-col gap-2 py-3 border-b border-surface-border last:border-0 sm:flex-row sm:items-start sm:justify-between"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <StatusBadge state={state} />
                  <span className="font-medium text-sm">{model.name}</span>
                </div>
                {model.ready ? (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm text-error"
                    onClick={() => handleDeleteClick(model)}
                  >
                    <Trash2 size={13} />
                    {t`Delete`}
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn btn-primary btn-sm whitespace-nowrap"
                    onClick={() => handleDownload(model.localName)}
                    disabled={!!downloading}
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
              {model.description && (
                <p className="text-sm text-content-subtle flex gap-1.5 items-start">
                  <Info size={12} className="mt-0.5 shrink-0" />
                  {model.description}
                </p>
              )}
              {progressText && <p className="text-sm text-content-subtle">{progressText}</p>}
            </div>
          );
        })}
      </SettingsSection>

      <dialog ref={dialogRef} className="modal">
        <div className="modal-box">
          {confirmDelete &&
            (() => {
              return (
                <>
                  <h3 className="font-bold text-lg">{t`Delete ${confirmDelete.name}?`}</h3>
                  {confirmDelete.description && (
                    <p className="py-2 text-sm text-content-subtle flex gap-2 items-start">
                      <AlertTriangle size={14} className="mt-0.5 shrink-0 text-warning" />
                      {confirmDelete.description}
                    </p>
                  )}
                  {confirmDelete.required && (
                    <p className="py-2 text-sm text-error flex gap-2 items-start">
                      <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                      {t`This is a required model. Dictation will not work without it.`}
                    </p>
                  )}
                  <div className="modal-action">
                    <button
                      type="button"
                      className="btn btn-ghost"
                      onClick={() => dialogRef.current?.close()}
                    >
                      {t`Cancel`}
                    </button>
                    <button type="button" className="btn btn-error" onClick={handleDeleteConfirm}>
                      {t`Delete model`}
                    </button>
                  </div>
                </>
              );
            })()}
          <button
            type="button"
            className="btn btn-sm btn-circle btn-ghost absolute right-2 top-2"
            onClick={() => dialogRef.current?.close()}
          >
            ✕
          </button>
        </div>
        <form method="dialog" className="modal-backdrop">
          <button type="submit">close</button>
        </form>
      </dialog>
    </>
  );
}
