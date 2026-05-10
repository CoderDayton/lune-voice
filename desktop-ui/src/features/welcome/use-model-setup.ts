import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react';
import { isTauriRuntime } from '../../lib/backend/runtime';

export interface ModelStatus {
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

export type SetupPhase = 'loading' | 'choose' | 'downloading' | 'error' | 'done';

const STORAGE_KEY = 'lune-setup-complete';

export function isSetupFlagSet(): boolean {
  return typeof window !== 'undefined' && window.localStorage?.getItem(STORAGE_KEY) === '1';
}

export function markSetupComplete(): void {
  window.localStorage?.setItem(STORAGE_KEY, '1');
}

interface FirstRunGate {
  status: 'checking' | 'needed' | 'ready';
  finish: () => void;
}

/**
 * Decides whether the welcome screen must be shown on app launch.
 * Re-verifies required models are present on disk even when the localStorage
 * flag is set — handles the case where a user deleted required models via
 * settings and re-launched.
 */
export function useFirstRunGate(): FirstRunGate {
  const [status, setStatus] = useState<'checking' | 'needed' | 'ready'>('checking');

  useEffect(() => {
    if (!isTauriRuntime()) {
      setStatus('ready');
      return;
    }
    let cancelled = false;
    invoke<ModelStatus[]>('check_model_status')
      .then((models) => {
        if (cancelled) return;
        const requiredReady = models.filter((m) => m.required).every((m) => m.ready);
        if (requiredReady && isSetupFlagSet()) {
          setStatus('ready');
        } else {
          setStatus('needed');
        }
      })
      .catch((err) => {
        if (cancelled) return;
        console.warn('[first-run-gate] check_model_status failed', err);
        setStatus('ready');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const finish = useCallback(() => {
    markSetupComplete();
    setStatus('ready');
  }, []);

  return { status, finish };
}

interface ModelSetup {
  phase: SetupPhase;
  models: ModelStatus[];
  selected: Set<string>;
  currentModel: string | null;
  progress: DownloadProgress | null;
  error: string | null;
  completed: number;
  totalToDownload: number;
  toggleModel: (localName: string) => void;
  startDownload: () => void;
  retry: () => void;
}

/**
 * Drives the model setup flow used on the welcome screen.
 * Owns the state machine: loading → choose → downloading → done | error.
 */
export function useModelSetup({ onFinish }: { onFinish: () => void }): ModelSetup {
  const [phase, setPhase] = useState<SetupPhase>('loading');
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [completed, setCompleted] = useState(0);
  const [totalToDownload, setTotalToDownload] = useState(0);

  const queueRef = useRef<string[]>([]);
  const currentRef = useRef<string | null>(null);

  useEffect(() => {
    currentRef.current = currentModel;
  }, [currentModel]);

  // Bootstrap: load model list
  useEffect(() => {
    if (!isTauriRuntime()) {
      onFinish();
      return;
    }
    let cancelled = false;
    invoke<ModelStatus[]>('check_model_status')
      .then((s) => {
        if (cancelled) return;
        setModels(s);
        const init = new Set<string>();
        for (const m of s) {
          if (m.ready || m.required) init.add(m.localName);
        }
        setSelected(init);
        const allReady = s.every((m) => m.ready);
        setPhase(allReady ? 'done' : 'choose');
      })
      .catch((err) => {
        if (cancelled) return;
        console.warn('[use-model-setup] check_model_status failed', err);
        onFinish();
      });
    return () => {
      cancelled = true;
    };
  }, [onFinish]);

  // Advance to next item in queue (or finish). Stable ref so the listener
  // below can call it without restarting subscriptions.
  const advance = useEffectEvent(() => {
    const next = queueRef.current.shift();
    if (!next) {
      setCurrentModel(null);
      setProgress(null);
      setPhase('done');
      return;
    }
    setCurrentModel(next);
    setProgress(null);
    invoke('download_model_cmd', { localName: next }).catch((err) => {
      setError(String(err));
      setCurrentModel(null);
      setPhase('error');
    });
  });

  // Subscribe to backend download events
  useEffect(() => {
    if (!isTauriRuntime()) return;
    const unlisten = listen<DownloadProgress>('lune://download-progress', (e) => {
      setProgress(e.payload);
      if (e.payload.done) {
        const finished = currentRef.current;
        if (finished) {
          setModels((prev) =>
            prev.map((m) => (m.localName === finished ? { ...m, ready: true } : m)),
          );
          setCompleted((c) => c + 1);
        }
        setProgress(null);
        advance();
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const toggleModel = useCallback(
    (localName: string) => {
      const m = models.find((x) => x.localName === localName);
      if (m?.required) return;
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(localName)) next.delete(localName);
        else next.add(localName);
        return next;
      });
    },
    [models],
  );

  const startDownload = useCallback(() => {
    const pending = models.filter((m) => selected.has(m.localName) && !m.ready);
    if (pending.length === 0) {
      setPhase('done');
      return;
    }
    queueRef.current = pending.map((m) => m.localName);
    setTotalToDownload(pending.length);
    setCompleted(0);
    setError(null);
    setPhase('downloading');
    advance();
  }, [models, selected]);

  const retry = useCallback(() => {
    if (currentRef.current) {
      // Re-queue the failed model and continue
      queueRef.current = [currentRef.current, ...queueRef.current];
    }
    setError(null);
    setPhase('downloading');
    advance();
  }, []);

  return {
    phase,
    models,
    selected,
    currentModel,
    progress,
    error,
    completed,
    totalToDownload,
    toggleModel,
    startDownload,
    retry,
  };
}
