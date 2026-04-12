import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Download, CircleAlert, ArrowRight, Mic, Shield, Wand2 } from "lucide-react";
import { useLingui } from "@lingui/react/macro";
import { isTauriRuntime } from "../../../lib/backend/runtime";
import { ModelCard } from "./model-card";

interface ModelStatus {
  name: string;
  repoId: string;
  localName: string;
  ready: boolean;
}

export interface DownloadProgress {
  modelName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  done: boolean;
}

type Phase = "checking" | "ready" | "setup" | "downloading" | "complete" | "error";

const MODEL_META: Record<string, { description: string; required: boolean; icon: typeof Mic }> = {
  "parakeet-tdt-0.6b": { description: "Speech recognition — converts voice to text. 25 languages with auto-detection.", required: true, icon: Mic },
  "nemotron-rerank-1b": { description: "Accuracy booster — rescores candidates to pick the most fluent transcription.", required: true, icon: Shield },
  "llama-3.2-1b-instruct-q4f16": { description: "Style formatting — reformats transcripts (casual, professional, code).", required: false, icon: Wand2 },
};

export function ModelSetup({ children }: { children: React.ReactNode }) {
  const { t } = useLingui();
  const [phase, setPhase] = useState<Phase>("checking");
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<string | null>(null);

  useEffect(() => {
    if (!isTauriRuntime()) { setPhase("ready"); return; }
    const setupDone = window.localStorage?.getItem("lune-setup-complete");
    invoke<ModelStatus[]>("check_model_status")
      .then((s) => {
        setModels(s);
        const requiredReady = s
          .filter((m) => MODEL_META[m.localName]?.required)
          .every((m) => m.ready);
        if (setupDone) setPhase("ready");
        else if (requiredReady) {
          // Required models present but user hasn't entered app yet — show
          // complete screen so they can still optionally download the LLM.
          const init = new Set<string>();
          for (const m of s) {
            if (m.ready || MODEL_META[m.localName]?.required) init.add(m.localName);
          }
          setSelected(init);
          setPhase("complete");
        } else {
          const init = new Set<string>();
          for (const m of s) {
            if (m.ready || MODEL_META[m.localName]?.required) init.add(m.localName);
          }
          setSelected(init);
          setPhase("setup");
        }
      })
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
            const next = s.find((m) => selected.has(m.localName) && !m.ready);
            if (next) {
              // Chain to next pending model
              setActiveModel(next.localName);
              setProgress(null);
              invoke("download_model_cmd", { localName: next.localName }).catch((e) => {
                setError(String(e)); setPhase("error"); setActiveModel(null);
              });
            } else {
              setPhase("complete");
              localStorage.setItem("lune-setup-complete", "1");
              invoke("initialize_pipeline").catch(() => {});
            }
          });
        }, 400);
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [activeModel, selected]);

  function toggle(localName: string) {
    if (MODEL_META[localName]?.required) return;
    setSelected((prev) => { const n = new Set(prev); n.has(localName) ? n.delete(localName) : n.add(localName); return n; });
  }

  function downloadSelected() {
    const pending = models.filter((x) => selected.has(x.localName) && !x.ready);
    if (pending.length === 0) return;
    const first = pending[0];
    setActiveModel(first.localName); setPhase("downloading"); setError(null); setProgress(null);
    // Fire-and-forget — progress events drive the UI, not the invoke Promise.
    // This prevents HMR reloads from orphaning the callback and showing false errors.
    invoke("download_model_cmd", { localName: first.localName }).catch((e) => {
      setError(String(e)); setPhase("error"); setActiveModel(null);
    });
  }

  if (phase === "ready") return <>{children}</>;
  if (phase === "checking") {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-base-100">
        <span className="loading loading-spinner loading-lg text-primary" />
      </div>
    );
  }

  const reqOk = models.filter((m) => MODEL_META[m.localName]?.required).every((m) => selected.has(m.localName));
  const toGet = models.filter((m) => selected.has(m.localName) && !m.ready);
  const pct = progress?.fileCount ? Math.round(((progress.fileIndex + 1) / progress.fileCount) * 100) : 0;
  const title = phase === "complete" ? t`You're all set!` : phase === "downloading" ? t`Downloading models` : phase === "error" ? t`Download failed` : t`Welcome to Lune Voice`;
  const subtitle = phase === "complete" ? t`All models are ready. Everything runs locally on your machine.`
    : phase === "downloading" ? t`This only happens once. Models are stored locally.`
    : phase === "error" ? error
    : t`Select the AI models to download. Everything runs offline — no cloud, no accounts.`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-base-100 p-4 sm:p-8">
      <div className="flex flex-col items-center gap-4 sm:gap-6 max-w-sm sm:max-w-lg w-full">
        <div className="text-center">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight">{title}</h2>
          <p className="mt-2 text-content-muted text-xs sm:text-sm max-w-xs sm:max-w-sm mx-auto">{subtitle}</p>
        </div>

        <div className="w-full space-y-2">
          {models.map((m) => {
            const meta = MODEL_META[m.localName];
            return (
              <ModelCard
                key={m.localName}
                name={m.name}
                localName={m.localName}
                description={meta?.description ?? m.repoId}
                required={meta?.required ?? false}
                icon={meta?.icon ?? Mic}
                ready={m.ready}
                selected={selected.has(m.localName)}
                active={activeModel === m.localName}
                canToggle={(phase === "setup" || phase === "complete") && !meta?.required && !m.ready}
                progress={activeModel === m.localName ? progress : null}
                pct={pct}
                onToggle={() => toggle(m.localName)}
              />
            );
          })}
        </div>

        {phase === "error" && (
          <div role="alert" className="alert alert-error alert-soft">
            <CircleAlert size={16} />
            <span className="text-sm">{error ?? t`Check your connection and try again.`}</span>
          </div>
        )}

        <div className="flex flex-col items-center gap-2 w-full">
          {phase === "complete" && toGet.length > 0 ? (
            <button type="button" onClick={downloadSelected} className="btn btn-primary btn-wide">
              <Download size={16} strokeWidth={2} />
              {t`Download ${toGet.length} optional model${toGet.length !== 1 ? "s" : ""}`}
            </button>
          ) : phase === "complete" ? (
            <button type="button" onClick={() => { localStorage.setItem("lune-setup-complete", "1"); setPhase("ready"); }} className="btn btn-success btn-wide">
              <ArrowRight size={16} strokeWidth={2} />{t`Enter Lune Voice`}
            </button>
          ) : (phase === "setup" || phase === "error") ? (
            <button type="button" onClick={downloadSelected} disabled={!reqOk || toGet.length === 0} className="btn btn-primary btn-wide">
              <Download size={16} strokeWidth={2} />
              {!reqOk ? t`Select required models` : t`Download ${toGet.length} model${toGet.length !== 1 ? "s" : ""}`}
            </button>
          ) : null}
          {phase === "downloading" ? (
            <p className="text-xs text-content-dim animate-pulse">{t`Please don't close the app during download`}</p>
          ) : (
            <button type="button" onClick={() => { localStorage.setItem("lune-setup-complete", "1"); setPhase("ready"); }} className="btn btn-ghost btn-sm text-content-dim">
              {t`Skip for now`}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
