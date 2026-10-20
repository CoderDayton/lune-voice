// SPDX-License-Identifier: GPL-3.0-or-later

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react';
import { backendClient } from '../../lib/backend/client';
import type { GpuBackendStatusDto } from '../../lib/backend/generated/GpuBackendStatusDto';
import type { GpuInstallProgressDto } from '../../lib/backend/generated/GpuInstallProgressDto';
import { installGpuBackend } from '../../lib/backend/gpu-install';
import { isTauriRuntime } from '../../lib/backend/runtime';
import {
  loadWorkspaceSnapshot,
  saveWorkspaceSetting,
  workspaceActions,
  workspaceStore,
} from '../../store/workspace-store';
import {
  type AsrBackendValue,
  languageForEngine,
  type ParakeetModelValue,
  systemLanguages,
  type WhisperModelValue,
} from '../settings/data/settings-data';
import {
  asrModelLocalName,
  buildInitialSelection,
  type GpuAccelerationOffer,
  gpuAccelerationOffer,
  hasOptionalModels,
  initialSetupEngine,
  initialWhisperModel,
  type ModelStatus,
  relevantSetupModels,
  shouldInstallGpuDuringSetup,
} from './setup-logic';

export type { ModelStatus };

export interface DownloadProgress {
  modelName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  done: boolean;
}

export type SetupPhase = 'loading' | 'engine' | 'choose' | 'downloading' | 'gpu' | 'error' | 'done';

const STORAGE_KEY = 'lune-setup-complete';

export function isSetupFlagSet(): boolean {
  return typeof window !== 'undefined' && window.localStorage?.getItem(STORAGE_KEY) === '1';
}

export function markSetupComplete(): void {
  window.localStorage?.setItem(STORAGE_KEY, '1');
}

/** Where an interrupted download queue is parked between window lifetimes. */
const QUEUE_KEY = 'lune-setup-queue';

/**
 * A download run, written down so closing the window does not lose it.
 *
 * Closing the main window destroys the webview and every ref in this hook with
 * it. The transfer itself keeps going — it runs on a blocking task in Rust — but
 * without this the models queued behind it were simply forgotten, and reopening
 * started setup again from the first step.
 */
interface SavedQueue {
  /** Models still to fetch, the one in flight first. */
  pending: string[];
  /** How many the run started with, so the progress bar keeps its scale. */
  total: number;
  wantsGpu: boolean;
}

function saveSetupQueue(queue: SavedQueue): void {
  try {
    window.localStorage?.setItem(QUEUE_KEY, JSON.stringify(queue));
  } catch {
    // Storage can be unavailable or full. Losing the resume record only costs
    // the user the clicks it would have saved, so it is never worth failing on.
  }
}

function loadSetupQueue(): SavedQueue | null {
  try {
    const raw = window.localStorage?.getItem(QUEUE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    // Hand-checked rather than trusted: this outlives the app version that
    // wrote it, and a shape change would otherwise crash the first screen.
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      !Array.isArray((parsed as SavedQueue).pending) ||
      !(parsed as SavedQueue).pending.every((n) => typeof n === 'string')
    ) {
      return null;
    }
    const queue = parsed as SavedQueue;
    return {
      pending: queue.pending,
      total: typeof queue.total === 'number' ? queue.total : queue.pending.length,
      wantsGpu: queue.wantsGpu === true,
    };
  } catch {
    return null;
  }
}

function clearSetupQueue(): void {
  try {
    window.localStorage?.removeItem(QUEUE_KEY);
  } catch {
    // Same reasoning as `saveSetupQueue`.
  }
}

interface FirstRunGate {
  status: 'checking' | 'needed' | 'ready';
  finish: () => void;
}

/**
 * Decides whether the welcome screen must be shown on app launch.
 *
 * Two conditions, and both matter. The localStorage flag alone would re-run
 * setup for someone who is already configured; the on-disk check alone would
 * never let a deliberately-skipped setup stay skipped. Together they mean the
 * screen appears exactly when the app cannot actually transcribe yet — after a
 * skipped setup, after deleting the model from Settings, or on a fresh profile.
 *
 * The readiness half asks the backend rather than deriving it from
 * `check_model_status`: what counts as needed depends on the configured ASR
 * backend (a remote one needs nothing local), and this is the same
 * `has_required_models` call the pipeline builds against, so the two cannot
 * drift apart.
 */
export function useFirstRunGate(): FirstRunGate {
  const [status, setStatus] = useState<'checking' | 'needed' | 'ready'>('checking');

  useEffect(() => {
    if (!isTauriRuntime()) {
      setStatus('ready');
      return;
    }
    let cancelled = false;
    invoke<boolean>('required_models_ready')
      .then((ready) => {
        if (cancelled) return;
        setStatus(ready && isSetupFlagSet() ? 'ready' : 'needed');
      })
      .catch((err) => {
        // Nothing is known about the models, so guessing "needed" would throw
        // setup at someone who finished it long ago. Let them into the app;
        // Settings › Models can still report and fix a missing download.
        if (cancelled) return;
        console.warn('[first-run-gate] required_models_ready failed', err);
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
  /** Chosen Parakeet size, used when engineBackend is ParakeetCpp. */
  engineParakeet: ParakeetModelValue;
  /**
   * Dictation language for the chosen engine. With Parakeet this is always a
   * language it covers — never Auto, which it cannot report back — starting
   * from the saved one, else the system's, else English.
   */
  engineLanguage: string;
  /** On-disk dir of the chosen engine's model — treated as required to finish. */
  asrLocalName: string;
  /** What the GPU acceptance card should offer, or `null` to hide it. */
  gpuOffer: GpuAccelerationOffer | null;
  /** Whether the user accepts GPU acceleration (mirrors the `useGpu` setting). */
  useGpu: boolean;
  /** Accept/decline GPU acceleration; persists via the shared settings path. */
  setUseGpu: (next: boolean) => void;
  willInstallGpu: boolean;
  gpuProgress: GpuInstallProgressDto | null;
  gpuDownloadBytes: number;
  /** The plugin was fetched during this run, so the GPU needs a restart. */
  gpuInstalled: boolean;
  /**
   * Why the plugin download failed, if it did. Never blocks setup: the app
   * runs on the CPU without it, and Settings can retry.
   */
  gpuError: string | null;
  setEngineBackend: (backend: AsrBackendValue) => void;
  setEngineWhisper: (model: WhisperModelValue) => void;
  setEngineParakeet: (model: ParakeetModelValue) => void;
  setEngineLanguage: (language: string) => void;
  confirmEngine: () => void;
  backToEngine: () => void;
  toggleModel: (localName: string) => void;
  startDownload: () => void;
  /** Stop the transfer in flight and return to the download step. */
  cancelDownload: () => void;
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
  const [engineParakeet, setEngineParakeet] = useState<ParakeetModelValue>('TdtV3Q8');
  // The saved (or picked) language; `engineLanguage` below adapts it to the
  // chosen engine without overwriting what the user had.
  const [pickedLanguage, setEngineLanguage] = useState<string>('auto');
  const engineLanguage = languageForEngine(engineBackend, pickedLanguage, systemLanguages());
  const [asrLocalName, setAsrLocalName] = useState<string>(() =>
    asrModelLocalName('WhisperCpp', 'Medium', 'TdtV3Q8'),
  );
  const [gpuOffer, setGpuOffer] = useState<GpuAccelerationOffer | null>(null);
  const [useGpu, setUseGpuState] = useState(true);
  const [gpuStatus, setGpuStatus] = useState<Pick<
    GpuBackendStatusDto,
    'state' | 'downloadAvailable' | 'downloadBytes'
  > | null>(null);
  const [gpuProgress, setGpuProgress] = useState<GpuInstallProgressDto | null>(null);
  const [gpuInstalled, setGpuInstalled] = useState(false);
  const [gpuError, setGpuError] = useState<string | null>(null);

  const queueRef = useRef<string[]>([]);
  const currentRef = useRef<string | null>(null);
  // The model whose download failed. Held separately from `currentModel`,
  // which the failure clears (and which only reaches `currentRef` on the next
  // render anyway) — reading it back in `retry` therefore found `null`, so the
  // failed model was silently dropped from the queue and setup could finish
  // reporting success with that model still missing.
  const failedRef = useRef<string | null>(null);
  // Latched when the download starts, so a model failure and its retry cannot
  // lose the pending GPU step, and a second pass cannot repeat it.
  const wantsGpuRef = useRef(false);
  // Set by `cancelDownload` so the rejection the backend answers with is
  // recognised as the user's own doing rather than reported as a failure.
  const cancelledRef = useRef(false);
  // Mirrors `totalToDownload` for the parked record. A run writes its total and
  // starts its first download in one tick, before that state has re-rendered.
  const totalRef = useRef(0);

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
        // Onboarding only offers on-device engines (the remote OpenAI option
        // lives in Settings); a saved Parakeet choice is kept.
        const backend = initialSetupEngine(cur.asrBackend);
        setEngineBackend(backend);
        setUseGpuState(cur.useGpu);
        // Runtime GPU state gates the acceptance card; treat a probe failure as
        // "no GPU" so a transient error never wrongly offers acceleration.
        const gpu = await backendClient.gpuBackendStatus().catch(() => ({
          state: 'unsupported' as const,
          downloadAvailable: false,
          downloadBytes: 0,
        }));
        if (cancelled) return;
        const offer = gpuAccelerationOffer(gpu.state, gpu.downloadAvailable);
        setGpuOffer(offer);
        setGpuStatus(gpu);
        const library = await invoke<ModelStatus[]>('check_model_status');
        if (cancelled) return;
        // First run: start the engine step on a size this machine can run.
        const whisper = initialWhisperModel(cur.whisperModel, library, {
          gpu: cur.useGpu && offer !== null,
          cpuThreads: navigator.hardwareConcurrency || 1,
        });
        setEngineWhisper(whisper);
        setEngineParakeet(cur.parakeetModel);
        setEngineLanguage(cur.dictationLanguage);
        const asrName = asrModelLocalName(backend, whisper, cur.parakeetModel);
        setAsrLocalName(asrName);
        const s = relevantSetupModels(library);
        setModels(s);
        setSelected(buildInitialSelection(asrName, s));

        // A run interrupted by the window closing picks up where it stopped
        // rather than sending the user back through the engine step.
        const active = await backendClient.activeModelDownload().catch(() => null);
        if (cancelled) return;
        if (resumeSetup(s, active)) return;

        // Models already on disk normally mean there is nothing to set up. A
        // pending GPU plugin is the exception: skipping to `done` would be the
        // user's only chance at the offer, spent silently.
        const gpuStepPending = shouldInstallGpuDuringSetup(
          gpu.state,
          gpu.downloadAvailable,
          cur.useGpu,
        );
        const nothingLeftToDo = s.length > 0 && s.every((m) => m.ready) && !gpuStepPending;
        setPhase(nothingLeftToDo ? 'done' : 'engine');
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

  // A failure here never fails setup. The app transcribes on the CPU without
  // the plugin, so the honest outcome is a finished setup that says the GPU
  // part did not happen — not a dead end in front of a first-run user.
  const runGpuInstall = useEffectEvent(() => {
    setPhase('gpu');
    setGpuError(null);
    setGpuProgress(null);
    installGpuBackend(setGpuProgress)
      .then((next) => {
        // Keep the status the rest of the flow reads in step with the install,
        // so nothing downstream still believes a download is pending.
        setGpuStatus(next);
        setGpuOffer(gpuAccelerationOffer(next.state, next.downloadAvailable));
        setGpuInstalled(true);
      })
      .catch((err) => {
        console.warn('[use-model-setup] GPU plugin install failed', err);
        setGpuError(String(err));
      })
      .finally(() => {
        setGpuProgress(null);
        setPhase('done');
      });
  });

  // Advance to next item in queue (or finish). Stable ref so the listener
  // below can call it without restarting subscriptions.
  const advance = useEffectEvent(() => {
    const next = queueRef.current.shift();
    if (!next) {
      setCurrentModel(null);
      setProgress(null);
      clearSetupQueue();
      if (wantsGpuRef.current) {
        wantsGpuRef.current = false;
        runGpuInstall();
        return;
      }
      setPhase('done');
      return;
    }
    setCurrentModel(next);
    setProgress(null);
    // Written down before the call, not after: the point of the record is to
    // survive the window closing, and the window can close at any moment
    // during the transfer this line is about to start.
    saveSetupQueue({
      pending: [next, ...queueRef.current],
      total: totalRef.current,
      wantsGpu: wantsGpuRef.current,
    });
    invoke('download_model_cmd', { localName: next }).catch((err) => {
      // A cancel rejects this call by design; reporting it as a failed download
      // would show an error for the thing the user just asked for.
      if (cancelledRef.current) return;
      failedRef.current = next;
      setError(String(err));
      setCurrentModel(null);
      setPhase('error');
    });
  });

  /**
   * Start a download run over `list`, honouring `selection` and the GPU intent.
   *
   * Takes its inputs as arguments rather than reading state so a caller that has
   * just fetched a fresh model list can act on it in the same tick, before React
   * has re-rendered with it.
   */
  const beginDownload = useEffectEvent(
    (list: ModelStatus[], selection: Set<string>, gpu: boolean) => {
      failedRef.current = null;
      cancelledRef.current = false;
      wantsGpuRef.current = gpu;

      const pending = list.filter((m) => selection.has(m.localName) && !m.ready);
      if (pending.length === 0) {
        clearSetupQueue();
        if (gpu) {
          wantsGpuRef.current = false;
          runGpuInstall();
          return;
        }
        setPhase('done');
        return;
      }

      queueRef.current = pending.map((m) => m.localName);
      totalRef.current = pending.length;
      setTotalToDownload(pending.length);
      setCompleted(0);
      setError(null);
      setPhase('downloading');
      advance();
    },
  );

  /**
   * Pick up a download run the last window lifetime left unfinished.
   *
   * Two sources, and the backend wins where they disagree: the parked queue says
   * what was still owed, and `active` says what is transferring right now. The
   * record can be a moment out of date — it is written just before each call —
   * so a model the backend is working on is always put at the head.
   *
   * Returns whether it took over the flow.
   */
  const resumeSetup = useEffectEvent((list: ModelStatus[], active: string | null): boolean => {
    const saved = loadSetupQueue();
    if (!saved && !active) return false;

    const isReady = (name: string) => list.some((m) => m.localName === name && m.ready);
    const remaining = (saved?.pending ?? []).filter((name) => !isReady(name));
    const queue = active ? [active, ...remaining.filter((n) => n !== active)] : remaining;

    if (queue.length === 0) {
      clearSetupQueue();
      return false;
    }

    // `total` only sets the progress bar's scale, so a record that disagrees
    // with reality must not produce a bar reading "3 of 2".
    const total = Math.max(saved?.total ?? queue.length, queue.length);
    wantsGpuRef.current = saved?.wantsGpu ?? false;
    failedRef.current = null;
    cancelledRef.current = false;
    totalRef.current = total;
    setTotalToDownload(total);
    setCompleted(total - queue.length);
    setError(null);
    setPhase('downloading');
    queueRef.current = queue;
    // Re-issuing the command for a model already in flight is safe: the backend
    // recognises the duplicate and answers from the transfer already running.
    advance();
    return true;
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
        // A file that lands in the same moment Cancel is pressed still counts as
        // installed, but must not carry the flow onward past the user's answer.
        if (cancelledRef.current) return;
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
      const asrName = asrModelLocalName(engineBackend, engineWhisper, engineParakeet);
      if (engineBackend === 'WhisperCpp') {
        // Stage the backend locally so the single save persists both keys in one
        // round trip (settingsToSaveInput serializes the whole store).
        workspaceActions.updateSetting('asrBackend', engineBackend);
        await saveWorkspaceSetting('whisperModel', engineWhisper);
      } else if (engineBackend === 'ParakeetCpp') {
        // Engine, size and language in one save: the backend refuses
        // Parakeet with Auto, so they cannot go separately.
        workspaceActions.updateSetting('asrBackend', engineBackend);
        workspaceActions.updateSetting('dictationLanguage', engineLanguage);
        await saveWorkspaceSetting('parakeetModel', engineParakeet);
      } else {
        await saveWorkspaceSetting('asrBackend', engineBackend);
      }
      setAsrLocalName(asrName);
      const s = relevantSetupModels(await invoke<ModelStatus[]>('check_model_status'));
      const selection = buildInitialSelection(asrName, s);
      setModels(s);
      setSelected(selection);
      // The choose step earns its click only when there is something to choose.
      // With one engine and one required model there never is: the list holds a
      // single locked card under a heading that says "choose", and every first
      // run pays a click to click past it.
      if (hasOptionalModels(asrName, s)) {
        setPhase('choose');
        return;
      }
      const gpu = gpuStatus
        ? shouldInstallGpuDuringSetup(gpuStatus.state, gpuStatus.downloadAvailable, useGpu)
        : false;
      beginDownload(s, selection, gpu);
    } catch (err) {
      console.error('[use-model-setup] confirmEngine failed', err);
      setError(String(err));
      setPhase('engine');
    }
  }, [engineBackend, engineWhisper, engineParakeet, engineLanguage, onFinish, gpuStatus, useGpu]);

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

  // Whether setup *has* a plugin to fetch, independent of the toggle. The
  // acceptance copy describes what accepting would do, so flipping the toggle
  // off must not rewrite it into a different situation.
  const gpuDownloadOffered = gpuStatus
    ? shouldInstallGpuDuringSetup(gpuStatus.state, gpuStatus.downloadAvailable, true)
    : false;
  const willInstallGpu = gpuDownloadOffered && useGpu;

  const startDownload = useCallback(() => {
    beginDownload(models, selected, willInstallGpu);
  }, [models, selected, willInstallGpu]);

  /**
   * Stop the transfer and go back to the download step.
   *
   * The backend notices within a chunk and drops the partial file, so nothing
   * half-written is left claiming to be a model. Returning to `choose` rather
   * than `error` matters: this is the user's own decision, and the screen they
   * land on is the one with the button that starts it again.
   */
  const cancelDownload = useCallback(() => {
    cancelledRef.current = true;
    queueRef.current = [];
    wantsGpuRef.current = false;
    clearSetupQueue();
    void backendClient.cancelModelDownload().catch((err) => {
      console.warn('[use-model-setup] cancel request failed', err);
    });
    setCurrentModel(null);
    setProgress(null);
    setError(null);
    setPhase('choose');
  }, []);

  const retry = useCallback(() => {
    cancelledRef.current = false;
    if (failedRef.current) {
      // Put the failed model back at the head of the queue: `advance` had
      // already shifted it off, so retrying without this would skip the one
      // download that did not work.
      queueRef.current = [failedRef.current, ...queueRef.current];
      failedRef.current = null;
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
    engineParakeet,
    engineLanguage,
    asrLocalName,
    gpuOffer,
    useGpu,
    setUseGpu,
    willInstallGpu,
    gpuProgress,
    gpuDownloadBytes: gpuStatus?.downloadBytes ?? 0,
    gpuInstalled,
    gpuError,
    setEngineBackend,
    setEngineWhisper,
    setEngineParakeet,
    setEngineLanguage,
    confirmEngine,
    backToEngine,
    toggleModel,
    startDownload,
    cancelDownload,
    retry,
  };
}
