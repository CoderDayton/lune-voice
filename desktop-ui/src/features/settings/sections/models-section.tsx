import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useStore } from "@tanstack/react-store";
import { SettingsSection } from "../components/section";
import { useLingui } from "@lingui/react/macro";
import { HardDrive, Download, Trash2, Loader2, AlertTriangle, Info, Zap, Circle } from "lucide-react";
import { workspaceStore } from "../../../store/workspace-store";

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
    description: "Speech-to-text engine (Parakeet TDT). Batch ASR across 25 languages.",
    required: false,
    deleteWarning: "Parakeet ASR backend will be unavailable. Switch to Qwen3 first if it's your active backend.",
  },
  "qwen3-asr-0.6b-onnx": {
    description: "Speech-to-text engine (Qwen3 ASR). Encoder-decoder ONNX model with SmoothQuant.",
    required: false,
    deleteWarning: "Qwen3 ASR backend will be unavailable. Switch to Parakeet first if it's your active backend.",
  },
  "nemotron-rerank-1b": {
    description: "Accuracy booster. Rescores transcription candidates to pick most fluent result.",
    required: true,
    deleteWarning: "Transcription accuracy will drop significantly. The reranker corrects common misrecognitions.",
  },
  "qwen3-0.6b": {
    description: "Style & correction LLM. Fixes grammar and reformats transcripts.",
    required: false,
    deleteWarning: "Style formatting and grammar correction will be disabled.",
  },
};

const ASR_MODEL_MAP: Record<string, string> = {
  ParakeetTdt: "parakeet-tdt-0.6b",
  Qwen3Asr: "qwen3-asr-0.6b-onnx",
};

function getModelState(localName: string, ready: boolean, activeAsrModel: string | undefined): "active" | "downloaded" | "missing" {
  if (!ready) return "missing";
  if (localName === activeAsrModel) return "active";
  if (localName === "nemotron-rerank-1b" || localName === "qwen3-0.6b") return "active";
  return "downloaded";
}

function StatusBadge({ state }: { state: "active" | "downloaded" | "missing" }) {
  switch (state) {
    case "active":
      return <span className="badge badge-sm badge-success gap-1"><Zap className="size-3" />Active</span>;
    case "downloaded":
      return <span className="badge badge-sm badge-ghost gap-1"><Circle className="size-3" />Idle</span>;
    case "missing":
      return <span className="badge badge-sm badge-warning gap-1"><AlertTriangle className="size-3" />Not installed</span>;
  }
}

export function ModelsSection() {
  const { t } = useLingui();
  const asrBackend = useStore(workspaceStore, (s) => s.settings.asrBackend);
  const activeAsrModel = ASR_MODEL_MAP[asrBackend];

  const [models, setModels] = useState<ModelStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<ModelStatus | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

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
      <SettingsSection title={t`Models`} description={t`Manage local model files`} icon={HardDrive}>
        <div className="flex flex-col gap-3">
          {models.map((model) => {
            const meta = MODEL_META[model.localName];
            const isDownloading = downloading === model.localName;
            const state = getModelState(model.localName, model.ready, activeAsrModel);
            const progressText = progress && isDownloading
              ? `${progress.fileName} (${progress.fileIndex + 1}/${progress.fileCount})`
              : null;

            return (
              <div key={model.localName} className="card card-border bg-base-200/50 card-sm">
                <div className="card-body gap-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{model.name}</span>
                      <StatusBadge state={state} />
                    </div>
                    <div className="flex gap-1">
                      {model.ready ? (
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs text-error"
                          onClick={() => handleDeleteClick(model)}
                        >
                          <Trash2 className="size-3.5" />
                          {t`Delete`}
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs text-primary"
                          onClick={() => handleDownload(model.localName)}
                          disabled={!!downloading}
                        >
                          {isDownloading ? (
                            <Loader2 className="size-3.5 animate-spin" />
                          ) : (
                            <Download className="size-3.5" />
                          )}
                          {isDownloading ? t`Downloading…` : t`Download`}
                        </button>
                      )}
                    </div>
                  </div>
                  {meta && (
                    <p className="text-xs text-base-content/60">{meta.description}</p>
                  )}
                  {progressText && (
                    <p className="text-xs text-info">{progressText}</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </SettingsSection>

      <dialog ref={dialogRef} className="modal">
        <div className="modal-box">
          {confirmDelete && (() => {
            const meta = MODEL_META[confirmDelete.localName];
            return (
              <>
                <h3 className="font-bold text-lg">{t`Delete ${confirmDelete.name}?`}</h3>
                {meta && (
                  <div className="alert alert-warning mt-4 text-sm">
                    <AlertTriangle className="size-4" />
                    <span>{meta.deleteWarning}</span>
                  </div>
                )}
                {meta?.required && (
                  <div className="alert alert-error mt-2 text-sm">
                    <Info className="size-4" />
                    <span>{t`This is a required model. Dictation will not work without it.`}</span>
                  </div>
                )}
                <div className="modal-action">
                  <button type="button" className="btn" onClick={() => dialogRef.current?.close()}>
                    {t`Cancel`}
                  </button>
                  <button type="button" className="btn btn-error" onClick={handleDeleteConfirm}>
                    {t`Delete model`}
                  </button>
                </div>
              </>
            );
          })()}
          <form method="dialog"><button type="submit" className="btn btn-sm btn-circle btn-ghost absolute right-2 top-2" aria-label="close">✕</button></form>
        </div>
      </dialog>
    </>
  );
}
