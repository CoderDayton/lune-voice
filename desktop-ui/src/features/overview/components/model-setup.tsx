import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Download, Check, Loader2, AlertCircle } from "lucide-react";
import { useLingui } from "@lingui/react/macro";
import { isTauriRuntime } from "../../../lib/backend/runtime";

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

type SetupPhase = "checking" | "ready" | "needs_models" | "downloading" | "error";

export function ModelSetup({ children }: { children: React.ReactNode }) {
  const { t } = useLingui();
  const [phase, setPhase] = useState<SetupPhase>("checking");
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);

  // Check model status on mount
  useEffect(() => {
    if (!isTauriRuntime()) {
      setPhase("ready");
      return;
    }

    invoke<ModelStatus[]>("check_model_status")
      .then((statuses) => {
        setModels(statuses);
        const allReady = statuses.every((m) => m.ready);
        setPhase(allReady ? "ready" : "needs_models");
      })
      .catch(() => setPhase("ready")); // If check fails, let the app try anyway
  }, []);

  // Listen for download progress
  useEffect(() => {
    if (!isTauriRuntime()) return;
    const unlisten = listen<DownloadProgress>("lune://download-progress", (e) => {
      setProgress(e.payload);
      if (e.payload.done) {
        setModels((prev) =>
          prev.map((m) =>
            m.localName === downloadingModel ? { ...m, ready: true } : m,
          ),
        );
        setDownloadingModel(null);
        // Check if all models are now ready
        setTimeout(() => {
          invoke<ModelStatus[]>("check_model_status").then((statuses) => {
            setModels(statuses);
            if (statuses.every((m) => m.ready)) {
              setPhase("ready");
            } else {
              setPhase("needs_models");
            }
          });
        }, 500);
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [downloadingModel]);

  async function handleDownload(localName: string) {
    setDownloadingModel(localName);
    setPhase("downloading");
    setError(null);
    setProgress(null);
    try {
      await invoke("download_model_cmd", { localName });
    } catch (e) {
      setError(String(e));
      setPhase("error");
      setDownloadingModel(null);
    }
  }

  async function handleDownloadAll() {
    const missing = models.filter((m) => !m.ready);
    for (const model of missing) {
      setDownloadingModel(model.localName);
      setPhase("downloading");
      setError(null);
      setProgress(null);
      try {
        await invoke("download_model_cmd", { localName: model.localName });
      } catch (e) {
        setError(String(e));
        setPhase("error");
        setDownloadingModel(null);
        return;
      }
    }
    setPhase("ready");
  }

  if (phase === "ready") return <>{children}</>;
  if (phase === "checking") {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <Loader2 size={24} className="animate-spin text-primary mb-4" />
        <p className="text-sm text-content-muted">{t`Checking models…`}</p>
      </div>
    );
  }

  const missing = models.filter((m) => !m.ready);

  return (
    <div className="page-stack">
      <section className="card border border-surface-border bg-surface-card">
        <div className="card-body items-center text-center gap-6 py-12 sm:py-16">
          {/* Icon */}
          <div className="flex size-16 items-center justify-center rounded-2xl bg-primary/10 sm:size-20">
            {phase === "downloading" ? (
              <Loader2 size={32} className="animate-spin text-primary" strokeWidth={1.5} />
            ) : phase === "error" ? (
              <AlertCircle size={32} className="text-error" strokeWidth={1.5} />
            ) : (
              <Download size={32} className="text-primary" strokeWidth={1.5} />
            )}
          </div>

          {/* Title */}
          <div>
            <h2 className="text-xl font-bold tracking-tight sm:text-2xl">
              {phase === "downloading"
                ? t`Downloading models`
                : phase === "error"
                  ? t`Download failed`
                  : t`Set up voice models`}
            </h2>
            <p className="mt-2 max-w-md text-sm text-content-muted">
              {phase === "downloading"
                ? t`This may take a few minutes depending on your connection. Models are stored locally and only downloaded once.`
                : phase === "error"
                  ? error
                  : t`Lune Voice needs to download speech recognition models to work offline. Everything runs locally on your machine.`}
            </p>
          </div>

          {/* Model list */}
          <div className="w-full max-w-sm space-y-2">
            {models.map((model) => (
              <div
                key={model.localName}
                className="flex items-center justify-between rounded-xl border border-surface-border bg-surface-inset px-4 py-3"
              >
                <div className="text-left">
                  <p className="text-sm font-medium">{model.name}</p>
                  <p className="text-xs text-content-subtle">{model.repoId}</p>
                </div>
                <div className="shrink-0">
                  {model.ready ? (
                    <Check size={18} className="text-success" strokeWidth={2.5} />
                  ) : downloadingModel === model.localName && progress ? (
                    <span className="text-xs tabular-nums text-primary font-medium">
                      {progress.fileIndex + 1}/{progress.fileCount}
                    </span>
                  ) : downloadingModel === model.localName ? (
                    <Loader2 size={16} className="animate-spin text-primary" />
                  ) : (
                    <button
                      type="button"
                      onClick={() => handleDownload(model.localName)}
                      className="btn btn-ghost btn-xs text-primary"
                      disabled={phase === "downloading"}
                    >
                      {t`Download`}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Progress bar */}
          {phase === "downloading" && progress && progress.fileCount > 0 && (
            <div className="w-full max-w-sm">
              <progress
                className="progress progress-primary w-full"
                value={progress.fileIndex + 1}
                max={progress.fileCount}
              />
              <p className="mt-1.5 text-xs text-content-subtle truncate">
                {progress.fileName || t`Preparing…`}
              </p>
            </div>
          )}

          {/* Actions */}
          {phase === "needs_models" && missing.length > 0 && (
            <button
              type="button"
              onClick={handleDownloadAll}
              className="btn btn-primary"
            >
              <Download size={16} strokeWidth={2} />
              {missing.length === 1
                ? t`Download model`
                : t`Download all models`}
            </button>
          )}

          {phase === "error" && (
            <button
              type="button"
              onClick={() => setPhase("needs_models")}
              className="btn btn-ghost"
            >
              {t`Try again`}
            </button>
          )}

          {/* Skip option */}
          {phase !== "downloading" && (
            <button
              type="button"
              onClick={() => setPhase("ready")}
              className="text-xs text-content-dim hover:text-content-subtle transition-colors"
            >
              {t`Skip for now`}
            </button>
          )}
        </div>
      </section>
    </div>
  );
}
