import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react';
import { backendClient } from '../../lib/backend/client';
import { isTauriRuntime } from '../../lib/backend/runtime';
import {
  loadWorkspaceSnapshot,
  saveWorkspaceSetting,
  workspaceActions,
  workspaceStore,
} from '../../store/workspace-store';
import type { AsrBackendValue, WhisperModelValue } from '../settings/data/settings-data';
import {
  asrModelLocalName,
  buildInitialSelection,
  type ModelStatus,
  relevantSetupModels,
  shouldOfferGpuAcceleration,
} from './setup-logic';

export type { ModelStatus };

export interface DownloadProgress {
  modelName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  done: boolean;
}

export type SetupPhase = 'loading' | 'engine' | 'choose' | 'downloading' | 'error' | 'done';

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
  /** Chosen ASR engine (engine step). */
  engineBackend: AsrBackendValue;
  /** Chosen Whisper size, used when engineBackend is WhisperCpp. */
  engineWhisper: WhisperModelValue;
  /** On-disk dir of the chosen engine's model — treated as required to finish. */
  asrLocalName: string;
  /** True when this machine could use an NVIDIA GPU — gates the acceptance card. */
  gpuAvailable: boolean;
  /** Whether the user accepts GPU acceleration (mirrors the `useGpu` setting). */
  useGpu: boolean;
  /** Accept/decline GPU acceleration; persists via the shared settings path. */
  setUseGpu: (next: boolean) => void;
  setEngineBackend: (backend: AsrBackendValue) => void;
  setEngineWhisper: (model: WhisperModelValue) => void;
  confirmEngine: () => void;
  backToEngine: () => void;
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
  const [engineBackend, setEngineBackend] = useState<AsrBackendValue>('WhisperCpp');
  const [engineWhisper, setEngineWhisper] = useState<WhisperModelValue>('Medium');
  const [asrLocalName, setAsrLocalName] = useState<string>(() =>
    asrModelLocalName('WhisperCpp', 'Medium'),
  );
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [useGpu, setUseGpuState] = useState(true);

  const queueRef = useRef<string[]>([]);
  const currentRef = useRef<string | null>(null);

  useEffect(() => {
    currentRef.current = currentModel;
  }, [currentModel]);

  // Bootstrap: hydrate real settings (the workspace bridge hasn't started yet
  // during onboarding) so the engine choice doesn't clobber them, pre-select the
  // current engine, then load model status and open the engine step.
  useEffect(() => {
    if (!isTauriRuntime()) {
      onFinish();
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        try {
          await loadWorkspaceSnapshot();
        } catch (err) {
          // Non-fatal: continue onboarding with default settings rather than
          // skipping it when the settings backend is briefly unavailable.
          console.warn('[use-model-setup] settings snapshot unavailable, using defaults', err);
        }
        if (cancelled) return;
        const cur = workspaceStore.state.settings;
        // Onboarding only offers on-device engines; whisper.cpp is the sole
        // local backend (the remote OpenAI option lives in Settings).
        const backend: AsrBackendValue = 'WhisperCpp';
        setEngineBackend(backend);
        setEngineWhisper(cur.whisperModel);
        const asrName = asrModelLocalName(backend, cur.whisperModel);
        setAsrLocalName(asrName);
        setUseGpuState(cur.useGpu);
        // Runtime GPU state gates the acceptance card; treat a probe failure as
        // "no GPU" so a transient error never wrongly offers acceleration.
        const gpuState = await backendClient
          .gpuBackendStatus()
          .then((s) => s.state)
          .catch(() => 'unsupported' as const);
        if (cancelled) return;
        setGpuAvailable(shouldOfferGpuAcceleration(gpuState));
        const s = relevantSetupModels(await invoke<ModelStatus[]>('check_model_status'));
        if (cancelled) return;
        setModels(s);
        setSelected(buildInitialSelection(asrName, s));
        setPhase(s.length > 0 && s.every((m) => m.ready) ? 'done' : 'engine');
      } catch (err) {
        if (cancelled) return;
        console.warn('[use-model-setup] bootstrap failed', err);
        onFinish();
      }
    })();
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
      // The chosen engine's model is required to finish setup.
      if (localName === asrLocalName) return;
      const m = models.find((x) => x.localName === localName);
      if (m?.required) return;
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(localName)) next.delete(localName);
        else next.add(localName);
        return next;
      });
    },
    [models, asrLocalName],
  );

  // Persist the chosen engine, then load its model list for the download step.
  const confirmEngine = useCallback(async () => {
    if (!isTauriRuntime()) {
      onFinish();
      return;
    }
    setPhase('loading');
    setError(null);
    try {
      const asrName = asrModelLocalName(engineBackend, engineWhisper);
      if (engineBackend === 'WhisperCpp') {
        // Stage the backend locally so the single save persists both keys in one
        // round trip (settingsToSaveInput serializes the whole store).
        workspaceActions.updateSetting('asrBackend', engineBackend);
        await saveWorkspaceSetting('whisperModel', engineWhisper);
      } else {
        await saveWorkspaceSetting('asrBackend', engineBackend);
      }
      setAsrLocalName(asrName);
      const s = relevantSetupModels(await invoke<ModelStatus[]>('check_model_status'));
      setModels(s);
      setSelected(buildInitialSelection(asrName, s));
      setPhase('choose');
    } catch (err) {
      console.error('[use-model-setup] confirmEngine failed', err);
      setError(String(err));
      setPhase('engine');
    }
  }, [engineBackend, engineWhisper, onFinish]);

  const backToEngine = useCallback(() => {
    setError(null);
    setPhase('engine');
  }, []);

  // Accept/decline GPU acceleration through the same settings path the
  // Transcription settings toggle uses, so the choice persists identically.
  const setUseGpu = useCallback((next: boolean) => {
    setUseGpuState(next);
    void saveWorkspaceSetting('useGpu', next);
  }, []);

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
    engineBackend,
    engineWhisper,
    asrLocalName,
    gpuAvailable,
    useGpu,
    setUseGpu,
    setEngineBackend,
    setEngineWhisper,
    confirmEngine,
    backToEngine,
    toggleModel,
    startDownload,
    retry,
  };
}
