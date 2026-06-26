import {
  type AsrBackendValue,
  QWEN3_ASR_LOCAL_NAME,
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
}

/**
 * On-disk model directory for a chosen engine + Whisper size. Pure so the
 * onboarding selection logic can be unit-tested without the Tauri/store deps.
 */
export function asrModelLocalName(backend: AsrBackendValue, whisper: WhisperModelValue): string {
  if (backend === 'WhisperCpp') {
    return whisperModels.find((m) => m.value === whisper)?.localName ?? QWEN3_ASR_LOCAL_NAME;
  }
  return QWEN3_ASR_LOCAL_NAME;
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
