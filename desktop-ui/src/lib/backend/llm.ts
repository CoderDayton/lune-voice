import { invoke } from '@tauri-apps/api/core';
import { isTauriRuntime } from './runtime';

/** Minimal shape of `check_model_status` entries needed for readiness. */
interface StyleModelStatus {
  role: string;
  ready: boolean;
  inUse: boolean;
}

export interface LlmReadiness {
  /** Whether the styling LLM can serve a request right now. */
  ready: boolean;
  /** User-facing reason to show when not ready; `null` when ready. */
  reason: string | null;
}

/** The subset of settings needed to judge styling-LLM readiness. */
export interface LlmSettings {
  llmEnabled: boolean;
  /** Which engine styling uses when enabled. */
  llmBackend: 'Local' | 'OpenAiCompatible';
  /** Remote provider config; the key is masked ('••••••••') when saved. */
  openaiLlmApiKey: string;
  openaiLlmModel: string;
}

const STYLING_OFF: LlmReadiness = {
  ready: false,
  reason: 'Styling is turned off. Enable a styling engine in Settings before generating examples.',
};

const MODEL_MISSING: LlmReadiness = {
  ready: false,
  reason:
    'The styling model isn’t downloaded yet. Download it in Settings → Models before generating examples.',
};

const PROVIDER_NOT_CONFIGURED: LlmReadiness = {
  ready: false,
  reason:
    'Your styling provider isn’t fully configured. Add its API key and model in Settings before generating examples.',
};

/**
 * Whether the styling LLM is ready to (re)generate few-shot examples: styling
 * must be enabled and, for the local engine, its model downloaded. Pass the
 * current `settings.llmEnabled` (the caller already holds it from the store, so
 * this module stays free of a store import cycle).
 *
 * Never throws — on any uncertainty (non-Tauri dev runtime, failed status
 * probe) it reports ready and lets the backend preflight have the final say.
 */
export async function getLlmReadiness(settings: LlmSettings): Promise<LlmReadiness> {
  if (!settings.llmEnabled) return STYLING_OFF;
  // Remote provider: gate on a configured API key + model (no local model).
  if (settings.llmBackend === 'OpenAiCompatible') {
    if (settings.openaiLlmApiKey.trim() === '' || settings.openaiLlmModel.trim() === '') {
      return PROVIDER_NOT_CONFIGURED;
    }
    return { ready: true, reason: null };
  }
  // Local engine: gate on the styling model being downloaded.
  if (!isTauriRuntime()) return { ready: true, reason: null };
  try {
    const models = await invoke<StyleModelStatus[]>('check_model_status');
    const style = models.find((m) => m.role === 'styling');
    if (style?.inUse && !style.ready) return MODEL_MISSING;
  } catch {
    return { ready: true, reason: null };
  }
  return { ready: true, reason: null };
}
