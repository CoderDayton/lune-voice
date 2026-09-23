// SPDX-License-Identifier: GPL-3.0-or-later

import type { GpuBackendStateDto } from '../../lib/backend/generated/GpuBackendStateDto';
import {
  type AsrBackendValue,
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
 * On-disk model directory for a chosen engine + Whisper size. Pure so the
 * onboarding selection logic can be unit-tested without the Tauri/store deps.
 */
export function asrModelLocalName(_backend: AsrBackendValue, whisper: WhisperModelValue): string {
  const fallback = whisperModels[0]?.localName ?? '';
  return whisperModels.find((m) => m.value === whisper)?.localName ?? fallback;
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
 * - `active` — a CUDA device is registered right now.
 * - `needs-restart` — the plugin is installed and only the next launch is
 *   missing. Nothing to download, nothing to fix.
 * - `after-download` — the hardware qualifies but the plugin still has to be
 *   fetched, which the setup download step does once the user accepts.
 *
 * Hidden entirely when no download could help: no native backend (`unsupported`),
 * no driver (`driver-missing`, and the driver is not redistributable), a driver
 * too old for the runtime (`failed`), or a build whose plugin archive has not
 * been published yet — offering setup the user cannot complete is worse than
 * staying quiet.
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

/**
 * Whether the setup download step should fetch the CUDA plugin.
 *
 * Only `not-installed` is worth a download: a plugin that is already on disk
 * is not made to work by fetching several hundred megabytes of it again,
 * whether it is waiting for a restart or refusing an out-of-date driver.
 */
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
