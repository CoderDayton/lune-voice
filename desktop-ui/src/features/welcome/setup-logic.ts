// SPDX-License-Identifier: GPL-3.0-or-later

import type { GpuBackendStateDto } from '../../lib/backend/generated/GpuBackendStateDto';
import {
  type AsrBackendValue,
  engineModelLocalName,
  type ParakeetModelValue,
  type WhisperModelValue,
  whisperModels,
} from '../settings/data/settings-data';

export interface ModelStatus {
  name: string;
  repoId: string;
  localName: string;
  ready: boolean;
  description: string;
  required: boolean;
  /** Whether the active backend selection currently uses this model. */
  inUse: boolean;
  /** Bytes a download would fetch, VAD included; `null` when unknown. */
  downloadBytes: number | null;
}

/**
 * Onboarding manages only the models the chosen configuration actually uses —
 * the selected Whisper size. `check_model_status` returns the full
 * installable library, so filter to the in-use entries to keep the setup
 * flow focused on what the user just chose.
 */
export function relevantSetupModels(models: ModelStatus[]): ModelStatus[] {
  return models.filter((m) => m.inUse);
}

/**
 * On-disk model directory for a chosen engine and its model. Pure so the
 * onboarding selection logic can be unit-tested without the Tauri/store deps.
 * Onboarding only offers on-device engines; anything else falls back to the
 * chosen Whisper size rather than leaving setup with nothing to install.
 */
export function asrModelLocalName(
  backend: AsrBackendValue,
  whisper: WhisperModelValue,
  parakeet: ParakeetModelValue,
): string {
  const fallback = whisperModels[0]?.localName ?? '';
  return (
    engineModelLocalName(backend, whisper, parakeet) ??
    engineModelLocalName('WhisperCpp', whisper, parakeet) ??
    fallback
  );
}

/**
 * The engine the setup flow starts on. A saved on-device choice stands, so
 * re-running setup never swaps a Parakeet user back to Whisper; a remote
 * engine (configured in Settings, never in onboarding) starts on Whisper.
 */
export function initialSetupEngine(saved: AsrBackendValue): AsrBackendValue {
  return saved === 'ParakeetCpp' ? 'ParakeetCpp' : 'WhisperCpp';
}

/**
 * CPU threads from which Small decodes a dictation in reasonable time without a
 * GPU. whisper.cpp uses at most 8 threads, so more does not change the answer.
 */
const SMALL_MODEL_MIN_CPU_THREADS = 8;

export interface SetupHardware {
  /** Whisper will decode on a GPU. */
  gpu: boolean;
  cpuThreads: number;
}

/**
 * The Whisper size the engine step starts on.
 *
 * On a first run — no Whisper model on disk yet — it is picked from the
 * hardware: Medium with a GPU, Small on a CPU with enough threads, Base below
 * that. Once any Whisper model is installed the saved choice stands, so
 * re-running setup never swaps the user's model. `models` is the full
 * `check_model_status` library, not the in-use subset.
 */
export function initialWhisperModel(
  saved: WhisperModelValue,
  models: ModelStatus[],
  hw: SetupHardware,
): WhisperModelValue {
  const whisperDirs = new Set(whisperModels.map((m) => m.localName));
  if (models.some((m) => m.ready && whisperDirs.has(m.localName))) return saved;
  if (hw.gpu) return 'Medium';
  return hw.cpuThreads >= SMALL_MODEL_MIN_CPU_THREADS ? 'Small' : 'Base';
}

/**
 * Models to pre-select for the download step: the chosen engine's own model
 * (always required to finish) plus any already-installed or backend-required
 * models.
 */
export function buildInitialSelection(asrLocalName: string, models: ModelStatus[]): Set<string> {
  const selected = new Set<string>([asrLocalName]);
  for (const m of models) {
    if (m.ready || m.required) selected.add(m.localName);
  }
  return selected;
}

/**
 * What the onboarding GPU-acceleration card should say, or `null` to hide it.
 *
 * The toggle sets the `useGpu` preference, which is meaningful before the
 * plugin exists — it decides whether the GPU gets used once one does. What is
 * not meaningful is claiming the machine is accelerated when it is not, so the
 * two cases are told apart:
 *
 * - `needs-restart` — the plugin is installed and only the next launch is
 *   missing. Nothing to download, nothing to fix.
 * - `after-download` — the hardware qualifies but the plugin still has to be
 *   fetched, which the setup download step does once the user accepts.
 */
export type GpuAccelerationOffer = 'active' | 'needs-restart' | 'after-download';

export function gpuAccelerationOffer(
  state: GpuBackendStateDto,
  downloadAvailable: boolean,
): GpuAccelerationOffer | null {
  if (state === 'ready') return 'active';
  if (state === 'needs-restart') return 'needs-restart';
  if (state !== 'not-installed') return null;
  return downloadAvailable ? 'after-download' : null;
}

export function shouldInstallGpuDuringSetup(
  state: GpuBackendStateDto,
  downloadAvailable: boolean,
  useGpu: boolean,
): boolean {
  return useGpu && downloadAvailable && state === 'not-installed';
}

/**
 * Whether the download step has a real choice in it — at least one model the
 * user could decline.
 *
 * The engine's own model is required to finish, and anything flagged `required`
 * cannot be turned off either, so with one engine and one Whisper size the list
 * is a single locked card. Presenting that under "Choose the AI models to
 * install" costs every first run a click and offers nothing back, so the step is
 * skipped when this is false.
 */
export function hasOptionalModels(asrLocalName: string, models: ModelStatus[]): boolean {
  return models.some((m) => m.localName !== asrLocalName && !m.required);
}

/**
 * Setup can proceed once the engine's model and every backend-required model
 * are selected (or already present).
 */
export function isSetupSatisfiable(
  asrLocalName: string,
  models: ModelStatus[],
  selected: Set<string>,
): boolean {
  return (
    selected.has(asrLocalName) &&
    models.filter((m) => m.required).every((m) => selected.has(m.localName))
  );
}

/**
 * Bytes a setup run would fetch: the selected models not yet on disk, plus
 * `gpuBytes` (0 when the GPU is not part of the run). `null` when a selected
 * model's size is unknown.
 */
export function setupDownloadBytes(
  models: ModelStatus[],
  selected: ReadonlySet<string>,
  gpuBytes: number,
): number | null {
  let sum = gpuBytes;
  for (const m of models) {
    if (!selected.has(m.localName) || m.ready) continue;
    if (m.downloadBytes === null) return null;
    sum += m.downloadBytes;
  }
  return sum;
}

/** A model in a download run, with its size fixed when the run started. */
export interface RunModel {
  localName: string;
  name: string;
  bytes: number;
}

export type GpuStep = 'queued' | 'downloading' | 'activating' | 'done' | 'failed';

export type ChecklistState = 'queued' | 'active' | 'activating' | 'done' | 'failed';

export interface ChecklistRow {
  /** The model's local name, or `gpu`. */
  id: string;
  /** The model's name; empty for the GPU row, which the UI labels. */
  name: string;
  state: ChecklistState;
  doneBytes: number;
  totalBytes: number;
}

export interface ChecklistInput {
  models: RunModel[];
  finished: ReadonlySet<string>;
  current: string | null;
  /** Bytes of `current` fetched so far. */
  currentBytes: number;
  failed: string | null;
  /**
   * Bytes of a model already on disk when its download stopped. Downloads
   * resume per file, so finished files stay and a retry fetches only the rest.
   */
  kept?: ReadonlyMap<string, number>;
  gpu: { bytes: number; step: GpuStep; downloadedBytes: number } | null;
}

/**
 * Every download in a setup run as one ordered list: the models in queue order,
 * then GPU support. The run downloads one item at a time, so at most one row is
 * active.
 */
export function buildChecklist(input: ChecklistInput): {
  rows: ChecklistRow[];
  doneBytes: number;
  totalBytes: number;
  /** No row is waiting or working. */
  settled: boolean;
} {
  const rows: ChecklistRow[] = input.models.map((m) => {
    const state: ChecklistState = input.finished.has(m.localName)
      ? 'done'
      : m.localName === input.failed
        ? 'failed'
        : m.localName === input.current
          ? 'active'
          : 'queued';
    const kept = input.kept?.get(m.localName) ?? 0;
    const doneBytes =
      state === 'done'
        ? m.bytes
        : state === 'active'
          ? Math.min(kept + input.currentBytes, m.bytes)
          : state === 'failed'
            ? Math.min(kept, m.bytes)
            : 0;
    return { id: m.localName, name: m.name, state, doneBytes, totalBytes: m.bytes };
  });

  if (input.gpu) {
    const { bytes, step, downloadedBytes } = input.gpu;
    const state: ChecklistState = step === 'downloading' ? 'active' : step;
    const doneBytes =
      step === 'activating' || step === 'done'
        ? bytes
        : step === 'downloading'
          ? Math.min(downloadedBytes, bytes)
          : 0;
    rows.push({ id: 'gpu', name: '', state, doneBytes, totalBytes: bytes });
  }

  // A failed GPU install is not retried by setup, so its bytes are owed by
  // nothing and would leave a finished setup short of 100%.
  const counted = rows.filter((r) => !(r.id === 'gpu' && r.state === 'failed'));
  return {
    rows,
    doneBytes: counted.reduce((sum, r) => sum + r.doneBytes, 0),
    totalBytes: counted.reduce((sum, r) => sum + r.totalBytes, 0),
    settled: rows.every((r) => r.state === 'done' || r.state === 'failed'),
  };
}
