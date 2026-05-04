import { useLingui } from '@lingui/react/macro';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { ArrowRight, CircleAlert, Download, Mic, Wand2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { isTauriRuntime } from '../../../lib/backend/runtime';
import { ModelCard } from './model-card';

interface ModelStatus {
  name: string;
  repoId: string;
  localName: string;
  ready: boolean;
  description: string;
  required: boolean;
}

export interface DownloadProgress {
  modelName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  done: boolean;
}

type Phase = 'checking' | 'ready' | 'setup' | 'downloading' | 'complete' | 'error';

function modelIcon(localName: string) {
  return localName.includes('asr') ? Mic : Wand2;
}

export function ModelSetup({ children }: { children: React.ReactNode }) {
  const { t } = useLingui();
  const [phase, setPhase] = useState<Phase>('checking');
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<string | null>(null);

  useEffect(() => {
    if (!isTauriRuntime()) {
      setPhase('ready');
      return;
    }
    const setupDone = window.localStorage?.getItem('lune-setup-complete');
    invoke<ModelStatus[]>('check_model_status')
      .then((s) => {
        setModels(s);
        const requiredReady = s.filter((m) => m.required).every((m) => m.ready);
        if (setupDone) setPhase('ready');
        else if (requiredReady) {
          const init = new Set<string>();
          for (const m of s) {
            if (m.ready || m.required) init.add(m.localName);
          }
          setSelected(init);
          setPhase('complete');
        } else {
          const init = new Set<string>();
          for (const m of s) {
            if (m.ready || m.required) init.add(m.localName);
          }
          setSelected(init);
          setPhase('setup');
        }
      })
      .catch(() => setPhase('ready'));
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    const unlisten = listen<DownloadProgress>('lune://download-progress', (e) => {
      setProgress(e.payload);
      if (e.payload.done) {
        setActiveModel((prev) => {
          setModels((models) =>
            models.map((m) => (m.localName === prev ? { ...m, ready: true } : m)),
          );
          return null;
        });
        setTimeout(() => {
          invoke<ModelStatus[]>('check_model_status').then((s) => {
            setModels(s);
            setSelected((sel) => {
              const next = s.find((m) => sel.has(m.localName) && !m.ready);
              if (next) {
                setActiveModel(next.localName);
                setProgress(null);
                invoke('download_model_cmd', { localName: next.localName }).catch((e) => {
                  setError(String(e));
                  setPhase('error');
                  setActiveModel(null);
                });
              } else {
                setPhase('complete');
                localStorage.setItem('lune-setup-complete', '1');
              }
              return sel;
            });
          });
        }, 400);
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  function toggle(localName: string) {
    const model = models.find((m) => m.localName === localName);
    if (model?.required) return;
    setSelected((prev) => {
      const n = new Set(prev);
      n.has(localName) ? n.delete(localName) : n.add(localName);
      return n;
    });
  }

  function downloadSelected() {
    const pending = models.filter((x) => selected.has(x.localName) && !x.ready);
    if (pending.length === 0) return;
    const first = pending[0];
    setActiveModel(first.localName);
    setPhase('downloading');
    setError(null);
    setProgress(null);
    invoke('download_model_cmd', { localName: first.localName }).catch((e) => {
      setError(String(e));
      setPhase('error');
      setActiveModel(null);
    });
  }

  if (phase === 'ready') return <>{children}</>;
  if (phase === 'checking') {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <span className="loading loading-spinner loading-lg" />
      </div>
    );
  }

  const reqOk = models.filter((m) => m.required).every((m) => selected.has(m.localName));
  const toGet = models.filter((m) => selected.has(m.localName) && !m.ready);
  const pct = progress?.fileCount
    ? Math.round(((progress.fileIndex + 1) / progress.fileCount) * 100)
    : 0;
  const title =
    phase === 'complete'
      ? t`You're all set!`
      : phase === 'downloading'
        ? t`Downloading models`
        : phase === 'error'
          ? t`Download failed`
          : t`Welcome to Lune Voice`;
  const subtitle =
    phase === 'complete'
      ? t`All models are ready. Everything runs locally on your machine.`
      : phase === 'downloading'
        ? t`This only happens once. Models are stored locally.`
        : phase === 'error'
          ? error
          : t`Select the AI models to download. Everything runs offline — no cloud, no accounts.`;

  return (
    <div className="flex min-h-screen items-center justify-center p-4 sm:p-8">
      <div className="w-full max-w-lg flex flex-col gap-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          <p className="text-sm text-content-subtle mt-2">{subtitle}</p>
        </div>

        <div className="flex flex-col gap-3">
          {models.map((m) => {
            return (
              <ModelCard
                key={m.localName}
                name={m.name}
                localName={m.localName}
                description={m.description}
                required={m.required}
                icon={modelIcon(m.localName)}
                ready={m.ready}
                selected={selected.has(m.localName)}
                active={activeModel === m.localName}
                canToggle={!m.required && phase !== 'downloading'}
                progress={activeModel === m.localName ? progress : null}
                pct={activeModel === m.localName ? pct : 0}
                onToggle={() => toggle(m.localName)}
              />
            );
          })}
        </div>

        {phase === 'error' && (
          <div className="alert alert-error">
            <CircleAlert size={16} />
            <span>{error ?? t`Check your connection and try again.`}</span>
          </div>
        )}

        <div className="flex flex-col items-center gap-2">
          {phase === 'complete' && toGet.length > 0 ? (
            <button type="button" className="btn btn-primary btn-wide" onClick={downloadSelected}>
              <Download size={16} />
              {t`Download ${toGet.length} optional model${toGet.length !== 1 ? 's' : ''}`}
            </button>
          ) : phase === 'complete' ? (
            <button
              type="button"
              className="btn btn-success btn-wide"
              onClick={() => {
                localStorage.setItem('lune-setup-complete', '1');
                setPhase('ready');
              }}
            >
              <ArrowRight size={16} />
              {t`Enter Lune Voice`}
            </button>
          ) : phase === 'setup' || phase === 'error' ? (
            <button
              type="button"
              className="btn btn-primary btn-wide"
              disabled={!reqOk}
              onClick={downloadSelected}
            >
              {!reqOk
                ? t`Select required models`
                : t`Download ${toGet.length} model${toGet.length !== 1 ? 's' : ''}`}
            </button>
          ) : null}
          {phase === 'downloading' ? (
            <p className="text-xs text-content-subtle">{t`Please don't close the app during download`}</p>
          ) : (
            <button
              type="button"
              className="btn btn-ghost btn-sm text-content-dim"
              onClick={() => {
                localStorage.setItem('lune-setup-complete', '1');
                setPhase('ready');
              }}
            >
              {t`Skip for now`}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
