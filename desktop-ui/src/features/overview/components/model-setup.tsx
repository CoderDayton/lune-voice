import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Download, Check, CircleAlert, Cpu, Sparkles } from "lucide-react";
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

type Phase = "checking" | "ready" | "setup" | "downloading" | "error";

export function ModelSetup({ children }: { children: React.ReactNode }) {
  const { t } = useLingui();
  const [phase, setPhase] = useState<Phase>("checking");
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<string | null>(null);

  useEffect(() => {
    if (!isTauriRuntime()) { setPhase("ready"); return; }
    invoke<ModelStatus[]>("check_model_status")
      .then((s) => { setModels(s); setPhase(s.every((m) => m.ready) ? "ready" : "setup"); })
      .catch(() => setPhase("ready"));
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    const unlisten = listen<DownloadProgress>("lune://download-progress", (e) => {
      setProgress(e.payload);
      if (e.payload.done) {
        setModels((prev) => prev.map((m) => m.localName === activeModel ? { ...m, ready: true } : m));
        setActiveModel(null);
        setTimeout(() => {
          invoke<ModelStatus[]>("check_model_status").then((s) => {
            setModels(s);
            setPhase(s.every((m) => m.ready) ? "ready" : "setup");
          });
        }, 400);
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [activeModel]);

  async function downloadAll() {
    for (const model of models.filter((m) => !m.ready)) {
      setActiveModel(model.localName);
      setPhase("downloading");
      setError(null);
      setProgress(null);
      try {
        await invoke("download_model_cmd", { localName: model.localName });
      } catch (e) {
        setError(String(e));
        setPhase("error");
        setActiveModel(null);
        return;
      }
    }
    setPhase("ready");
  }

  if (phase === "ready") return <>{children}</>;

  if (phase === "checking") {
    return (
      <div className="hero min-h-[60vh]">
        <div className="hero-content text-center">
          <span className="loading loading-spinner loading-lg text-primary" />
        </div>
      </div>
    );
  }

  const missing = models.filter((m) => !m.ready);
  const pct = progress && progress.fileCount > 0
    ? Math.round(((progress.fileIndex + 1) / progress.fileCount) * 100)
    : 0;

  return (
    <div className="hero min-h-[60vh]">
      <div className="hero-content flex-col gap-6 sm:gap-8 max-w-sm sm:max-w-lg w-full px-4 sm:px-0">

        {/* Header */}
        <div className="text-center">
          <div className="mb-5 flex justify-center">
            <div className="radial-progress text-primary" style={{ "--value": phase === "downloading" ? pct : 0, "--size": "5rem", "--thickness": "4px" } as React.CSSProperties} role="progressbar" aria-valuenow={pct}>
              {phase === "downloading" ? (
                <span className="text-lg font-bold tabular-nums">{pct}%</span>
              ) : phase === "error" ? (
                <CircleAlert size={28} strokeWidth={1.5} className="text-error" />
              ) : (
                <Sparkles size={28} strokeWidth={1.5} className="text-primary" />
              )}
            </div>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight">
            {phase === "downloading" ? t`Downloading` : phase === "error" ? t`Download failed` : t`Welcome to Lune Voice`}
          </h2>
          <p className="mt-2 text-content-muted text-xs sm:text-sm max-w-xs sm:max-w-sm mx-auto">
            {phase === "downloading"
              ? t`Models are stored locally and only downloaded once.`
              : phase === "error"
                ? error
                : t`Download the AI models that power offline voice recognition. Everything runs on your machine — no cloud, no accounts.`}
          </p>
        </div>

        {/* Model cards */}
        <div className="w-full space-y-2">
          {models.map((model) => {
            const isActive = activeModel === model.localName;
            const Icon = model.localName.includes("llm") ? Sparkles : Cpu;
            return (
              <div key={model.localName} className="card card-border bg-base-200 card-sm">
                <div className="card-body flex-row items-center gap-2 sm:gap-3">
                  <div className={`flex size-10 shrink-0 items-center justify-center rounded-xl ${model.ready ? "bg-success/10" : isActive ? "bg-primary/10" : "bg-base-300"}`}>
                    {model.ready ? (
                      <Check size={18} strokeWidth={2.5} className="text-success" />
                    ) : isActive ? (
                      <span className="loading loading-spinner loading-sm text-primary" />
                    ) : (
                      <Icon size={18} strokeWidth={1.5} className="text-content-subtle" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm">{model.name}</p>
                    {isActive && progress ? (
                      <div className="mt-1">
                        <progress className="progress progress-primary h-1.5 w-full" value={progress.fileIndex + 1} max={progress.fileCount} />
                        <p className="text-xs text-content-dim mt-0.5 truncate">{progress.fileName}</p>
                      </div>
                    ) : (
                      <p className="text-xs text-content-subtle truncate">{model.repoId}</p>
                    )}
                  </div>
                  <div className="shrink-0">
                    {model.ready ? (
                      <div className="badge badge-soft badge-success badge-sm">{t`Ready`}</div>
                    ) : isActive ? (
                      <span className="text-xs tabular-nums text-primary font-medium">{progress ? `${progress.fileIndex + 1}/${progress.fileCount}` : "…"}</span>
                    ) : null}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Error alert */}
        {phase === "error" && (
          <div role="alert" className="alert alert-error alert-soft">
            <CircleAlert size={16} />
            <span className="text-sm">{t`Check your internet connection and try again.`}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex flex-col items-center gap-2 sm:gap-3 w-full">
          {(phase === "setup" || phase === "error") && (
            <button type="button" onClick={downloadAll} className="btn btn-primary btn-wide">
              <Download size={16} strokeWidth={2} />
              {missing.length === 1 ? t`Download model` : t`Download ${missing.length} models`}
            </button>
          )}
          {phase !== "downloading" && (
            <button type="button" onClick={() => setPhase("ready")} className="btn btn-ghost btn-sm text-content-dim">
              {t`Skip for now`}
            </button>
          )}
          {phase === "downloading" && (
            <p className="text-xs text-content-dim animate-pulse">{t`Please don't close the app during download`}</p>
          )}
        </div>

      </div>
    </div>
  );
}
