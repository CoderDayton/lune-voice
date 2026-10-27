// SPDX-License-Identifier: GPL-3.0-or-later

import { invoke } from '@tauri-apps/api/core';
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
import type { GpuStep, RunChecklist, RunModel } from '../model-downloads/checklist';
import { runChecklist } from '../model-downloads/download-run';
import { modelDownloads, useModelDownloads } from '../model-downloads/model-downloads';
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
  isWizardStep,
  type ModelStatus,
  relevantSetupModels,
  shouldInstallGpuDuringSetup,
  stepAfter,
  stepBefore,
  type WizardStep,
  wizardSteps,
} from './setup-logic';

export type { ModelStatus };

/**
 * The wizard's screens, then the run itself. `gpu` is the plugin install, not
 * the screen that offers it — that one is the `acceleration` step.
 */
export type SetupPhase = 'loading' | WizardStep | 'downloading' | 'gpu' | 'error' | 'done';

const STORAGE_KEY = 'lune-setup-complete';

export function isSetupFlagSet(): boolean {
  return typeof window !== 'undefined' && window.localStorage?.getItem(STORAGE_KEY) === '1';
}

export function markSetupComplete(): void {
  window.localStorage?.setItem(STORAGE_KEY, '1');
}

function toRunModel(m: ModelStatus): RunModel {
  return { localName: m.localName, name: m.name, bytes: m.downloadBytes ?? 0 };
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
  error: string | null;
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
  /** The downloads of the current run; `null` before a run starts. */
  checklist: RunChecklist | null;
  setEngineBackend: (backend: AsrBackendValue) => void;
  setEngineWhisper: (model: WhisperModelValue) => void;
  setEngineParakeet: (model: ParakeetModelValue) => void;
  setEngineLanguage: (language: string) => void;
  /** The wizard's screens for the current choices, in order. */
  steps: WizardStep[];
  confirmEngine: () => void;
  /** Move to the next wizard screen. */
  goNext: () => void;
  /** Move to the previous wizard screen. */
  goBack: () => void;
  toggleModel: (localName: string) => void;
  startDownload: () => void;
  /** Stop the transfer in flight and return to the review. */
  cancelDownload: () => void;
  retry: () => void;
}

/**
 * Drives the model setup flow used on the welcome screen.
 * Owns the state machine: loading → the wizard screens (engine → acceleration →
 * extras → review) → downloading → done | error.
 */
export function useModelSetup({ onFinish }: { onFinish: () => void }): ModelSetup {
  const [phase, setPhase] = useState<SetupPhase>('loading');
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // `models` describes the engine being picked only once the engine step has
  // saved the choice and reloaded the list.
  const [modelsConfirmed, setModelsConfirmed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = useModelDownloads();
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
  const [gpuStep, setGpuStep] = useState<GpuStep | null>(null);

  // Latched when the download starts, so a model failure and its retry cannot
  // lose the pending GPU step, and a second pass cannot repeat it.
  const wantsGpuRef = useRef(false);

  // A finished or failed run is this wizard's, so it must not keep showing as a
  // checklist on the Models page once setup is left.
  useEffect(() => () => modelDownloads.dismiss(), []);

  // The queue decides when the models are in or one failed; this decides what
  // the wizard does next.
  useEffect(() => {
    if (phase !== 'downloading') return;
    if (run.status === 'settled') {
      if (wantsGpuRef.current) {
        wantsGpuRef.current = false;
        runGpuInstall();
        return;
      }
      setPhase('done');
    } else if (run.status === 'failed') {
      setPhase('error');
    }
  }, [phase, run.status]);

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
        if (resumeSetup(library, active)) return;

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
    setGpuStep('downloading');
    installGpuBackend((p) => {
      setGpuProgress(p);
      // `done` arrives once the archive is unpacked. The install then loads the
      // plugin and moves the model onto the GPU before it resolves.
      if (p.done) setGpuStep('activating');
    })
      .then((next) => {
        // Keep the status the rest of the flow reads in step with the install,
        // so nothing downstream still believes a download is pending.
        setGpuStatus(next);
        setGpuOffer(gpuAccelerationOffer(next.state, next.downloadAvailable));
        // The plugin loaded but found no usable GPU. A restart would not
        // change that, so show the backend's reason instead.
        if (next.state === 'failed') {
          setGpuError(next.reason);
          setGpuStep('failed');
          return;
        }
        setGpuInstalled(true);
        setGpuStep('done');
      })
      .catch((err) => {
        console.warn('[use-model-setup] GPU plugin install failed', err);
        setGpuError(String(err));
        setGpuStep('failed');
      })
      .finally(() => {
        setGpuProgress(null);
        // Only finish if the install still owns the screen: a user who has
        // since left the phase must not be pulled back to the end.
        setPhase((current) => (current === 'gpu' ? 'done' : current));
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
      wantsGpuRef.current = gpu;
      const pending = list.filter((m) => selection.has(m.localName) && !m.ready);
      setGpuStep(gpu ? 'queued' : null);
      if (pending.length === 0) {
        if (gpu) {
          wantsGpuRef.current = false;
          runGpuInstall();
          return;
        }
        setPhase('done');
        return;
      }

      setError(null);
      modelDownloads.start(pending.map(toRunModel), gpu ? 'gpu' : null);
      setPhase('downloading');
    },
  );

  const resumeSetup = useEffectEvent((library: ModelStatus[], active: string | null): boolean => {
    const resumed = modelDownloads.resume(
      library.map((m) => ({ ...toRunModel(m), ready: m.ready })),
      active,
    );
    if (!resumed) return false;
    wantsGpuRef.current = resumed.tag === 'gpu';
    setGpuStep(wantsGpuRef.current ? 'queued' : null);
    setError(null);
    setPhase('downloading');
    return true;
  });

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
      setModelsConfirmed(true);
      // The list of screens depends on the model list just fetched, so it is
      // worked out here from that rather than from state React has not yet
      // re-rendered with. Nothing downloads until the review screen.
      setPhase(stepAfter(wizardSteps(gpuOffer, hasOptionalModels(asrName, s)), 'engine'));
    } catch (err) {
      console.error('[use-model-setup] confirmEngine failed', err);
      setError(String(err));
      setPhase('engine');
    }
  }, [engineBackend, engineWhisper, engineParakeet, engineLanguage, onFinish, gpuOffer]);

  // Before the engine is confirmed `models` is the saved choice's list, not the
  // one being picked, so it says nothing about extras yet.
  const pickEngine =
    <T>(set: (value: T) => void) =>
    (value: T) => {
      set(value);
      setModelsConfirmed(false);
    };
  const steps = wizardSteps(gpuOffer, modelsConfirmed && hasOptionalModels(asrLocalName, models));

  const goNext = useCallback(() => {
    setError(null);
    setPhase((current) => (isWizardStep(current) ? stepAfter(steps, current) : current));
  }, [steps]);

  const goBack = useCallback(() => {
    setError(null);
    setPhase((current) => (isWizardStep(current) ? stepBefore(steps, current) : current));
  }, [steps]);

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

  // Finished by the shared queue means installed, whatever the earlier list says.
  const modelsNow = models.map((m) =>
    run.finished.includes(m.localName) ? { ...m, ready: true } : m,
  );

  const checklist: RunChecklist | null =
    run.models.length > 0 || gpuStep !== null
      ? runChecklist(
          run,
          gpuStep === null
            ? null
            : {
                bytes: gpuStatus?.downloadBytes ?? 0,
                step: gpuStep,
                downloadedBytes: gpuProgress?.downloadedBytes ?? 0,
              },
        )
      : null;

  const startDownload = useCallback(() => {
    beginDownload(models, selected, willInstallGpu);
  }, [models, selected, willInstallGpu]);

  /**
   * Stop the transfer and go back to the download step.
   *
   * The backend notices within a chunk and drops the partial file, so nothing
   * half-written is left claiming to be a model. Returning to `choose` rather
   * than `error` matters: this is the user's own decision, and the screen they
   * land on is the review, with the button that starts it again.
   */
  const cancelDownload = useCallback(() => {
    modelDownloads.cancel();
    wantsGpuRef.current = false;
    setGpuStep(null);
    setError(null);
    setPhase('review');
  }, []);

  const retry = useCallback(() => {
    setError(null);
    modelDownloads.retry();
    setPhase('downloading');
  }, []);

  return {
    phase,
    models: modelsNow,
    selected,
    error: run.status === 'failed' ? run.error : error,
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
    checklist,
    setEngineBackend: pickEngine(setEngineBackend),
    setEngineWhisper: pickEngine(setEngineWhisper),
    setEngineParakeet: pickEngine(setEngineParakeet),
    setEngineLanguage: pickEngine(setEngineLanguage),
    steps,
    confirmEngine,
    goNext,
    goBack,
    toggleModel,
    startDownload,
    cancelDownload,
    retry,
  };
}
