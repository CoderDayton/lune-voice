import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { SettingsSection } from "../components/section";
import { useLingui } from "@lingui/react/macro";
import { HardDrive, Download, Trash2, Loader2, CheckCircle2, AlertTriangle, Info } from "lucide-react";

interface ModelStatus {
  name: string;
  repoId: string;
  localName: string;
  ready: boolean;
}

interface DownloadProgress {
  modelName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  done: boolean;
}

interface ModelMeta {
  description: string;
  required: boolean;
  deleteWarning: string;
}

const MODEL_META: Record<string, ModelMeta> = {
  "parakeet-tdt-0.6b": {
    description: "Core speech-to-text engine. Converts voice to text across 25 languages with automatic detection.",
    required: true,
    deleteWarning: "Speech recognition will stop working. You won't be able to dictate until this model is re-downloaded.",
  },
  "nemotron-rerank-1b": {
    description: "Accuracy booster. Rescores transcription candidates to pick the most fluent result.",
    required: true,
    deleteWarning: "Transcription accuracy will drop significantly. The reranker corrects common misrecognitions.",
  },
  "llama-3.2-1b-instruct-q4f16": {
    description: "Style formatter. Reformats transcripts into casual, professional, or technical tone.",
    required: false,
    deleteWarning: "Style formatting will be disabled. Transcripts will use basic text normalization only.",
  },
};

export function ModelsSection() {
  const { t } = useLingui();
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<ModelStatus | null>(null);
  const dialogRef = useRef<HTMLDialogElement | null>(null);

  function refresh() {
    invoke<ModelStatus[]>("check_model_status")
      .then(setModels)
      .catch((e) => console.error("check_model_status:", e))
      .finally(() => setLoading(false));
  }

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    const unlisten = listen<DownloadProgress>("lune://download-progress", (event) => {
      setProgress(event.payload);
      if (event.payload.done) {
        setDownloading(null);
        setProgress(null);
        refresh();
      }
    });
    return () => { void unlisten.then((f) => f()); };
  }, []);

  function handleDownload(localName: string) {
    setDownloading(localName);
    invoke("download_model_cmd", { localName }).catch((e) => {
      console.error("download_model_cmd:", e);
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
    invoke("delete_model_cmd", { localName })
      .then(() => refresh())
      .catch((e) => console.error("delete_model_cmd:", e));
  }

  if (loading) return null;

  return (
    <>
    <SettingsSection
      title={t`Models`}
      description={t`Download or remove local AI models. Models are stored on disk and used for speech recognition and text processing.`}
      icon={HardDrive}
    >
      <div className="flex flex-col gap-1">
        {models.map((model) => {
          const meta = MODEL_META[model.localName];
          const isDownloading = downloading === model.localName;
          const progressText = progress && isDownloading
            ? `${progress.fileName} (${progress.fileIndex + 1}/${progress.fileCount})`
            : null;

          return (
            <div
              key={model.localName}
              className="flex items-start justify-between gap-4 py-3 [&:not(:last-child)]:border-b [&:not(:last-child)]:border-surface-border"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium">{model.name}</span>
                  {model.ready && (
                    <CheckCircle2 size={14} className="text-success shrink-0" />
                  )}
                  {meta?.required && (
                    <div className="tooltip tooltip-top" data-tip="This model is required for core functionality">
                      <span className="badge badge-warning badge-xs gap-1 font-medium">
                        Required
                      </span>
                    </div>
                  )}
                </div>
                <p className="text-xs text-content-subtle mt-0.5">
                  {isDownloading && progressText
                    ? progressText
                    : meta?.description ?? model.repoId}
                </p>
              </div>
              <div className="shrink-0 pt-0.5">
                {model.ready ? (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm text-error"
                    onClick={() => handleDeleteClick(model)}
                    disabled={isDownloading}
                  >
                    <Trash2 size={14} />
                    {t`Delete`}
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => handleDownload(model.localName)}
                    disabled={!!downloading}
                  >
                    {isDownloading ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Download size={14} />
                    )}
                    {isDownloading ? t`Downloading…` : t`Download`}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </SettingsSection>

    <dialog ref={dialogRef} className="modal backdrop-blur-sm" onClose={() => setConfirmDelete(null)}>
      {confirmDelete && (() => {
        const meta = MODEL_META[confirmDelete.localName];
        return (
          <div className="modal-box max-w-sm border border-surface-border">
            <div className="flex items-start gap-3">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-error/10 mt-0.5">
                <AlertTriangle size={18} className="text-error" />
              </div>
              <div>
                <h3 className="text-base font-semibold">{t`Delete ${confirmDelete.name}?`}</h3>
                <p className="text-sm text-content-muted mt-2">{meta?.deleteWarning}</p>
                {meta?.required && (
                  <div className="flex items-start gap-2 mt-3 rounded-lg bg-warning/8 border border-warning/20 p-3">
                    <Info size={14} className="text-warning shrink-0 mt-0.5" />
                    <p className="text-xs text-warning">{t`This is a required model. Dictation will not work without it.`}</p>
                  </div>
                )}
              </div>
            </div>
            <div className="modal-action">
              <form method="dialog">
                <button type="submit" className="btn btn-ghost btn-sm">{t`Cancel`}</button>
              </form>
              <button type="button" className="btn btn-error btn-sm" onClick={handleDeleteConfirm}>
                <Trash2 size={14} />
                {t`Delete model`}
              </button>
            </div>
          </div>
        );
      })()}
      <form method="dialog" className="modal-backdrop"><button type="submit">close</button></form>
    </dialog>
    </>
  );
}
