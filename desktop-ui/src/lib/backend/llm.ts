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
/**
 * The configuration checks that need no backend round-trip: styling disabled,
 * or a remote provider missing its key/model. Returns a not-ready reason, or
 * `null` when configuration is fine and only the (async) engine/model state is
 * left to decide. Shared by `getLlmReadiness` and `ensureLlmInitialized` so
 * neither pays a redundant IPC for these cases.
 */
function configReason(settings: LlmSettings): LlmReadiness | null {
  if (!settings.llmEnabled) return STYLING_OFF;
  if (
    settings.llmBackend === 'OpenAiCompatible' &&
    (settings.openaiLlmApiKey.trim() === '' || settings.openaiLlmModel.trim() === '')
  ) {
    return PROVIDER_NOT_CONFIGURED;
  }
  return null;
}

export async function getLlmReadiness(settings: LlmSettings): Promise<LlmReadiness> {
  const configIssue = configReason(settings);
  if (configIssue) return configIssue;
  // Remote provider needs no local model — configuration is the whole check.
  if (settings.llmBackend === 'OpenAiCompatible') return { ready: true, reason: null };
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

/**
 * Ensure the styling engine is fully initialized — provider configured AND,
 * for the local engine, the model actually loaded into memory and able to
 * serve. Unlike `getLlmReadiness` (a cheap configuration check), this blocks
 * until the load completes (seconds on a cold engine), so callers must show a
 * busy state while awaiting it. Never throws.
 */
export async function ensureLlmInitialized(settings: LlmSettings): Promise<LlmReadiness> {
  // The cheap, IPC-free config checks first — a disabled or unconfigured
  // provider needs no round-trip and gets a friendly message.
  const configIssue = configReason(settings);
  if (configIssue) return configIssue;
  if (!isTauriRuntime()) return { ready: true, reason: null };
  // `warm_up_styling_llm` already re-checks configuration and validates the
  // local model end-to-end (loading it into memory), so the separate
  // `getLlmReadiness` model probe would be a duplicate round-trip — skip it and
  // let warm-up be the single authority.
  try {
    await invoke('warm_up_styling_llm');
    return { ready: true, reason: null };
  } catch (e) {
    return { ready: false, reason: String(e) };
  }
}
