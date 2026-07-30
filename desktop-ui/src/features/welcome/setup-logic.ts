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
 * Whether to surface the GPU-acceleration acceptance card during onboarding.
 * GPU support is a build-time decision (CUDA is linked only in a CUDA build),
 * so the card appears only when the backend reports a compiled-in CUDA feature;
 * otherwise the onboarding flow is unchanged.
 */
export function shouldOfferGpuAcceleration(gpuCompiledIn: boolean): boolean {
  return gpuCompiledIn;
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
