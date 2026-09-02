import type { MessageDescriptor } from '@lingui/core';
import { msg } from '@lingui/core/macro';

export interface LlmReadiness {
  /** Whether the styling LLM can serve a request right now. */
  ready: boolean;
  /** User-facing reason to show when not ready; `null` when ready. Translated
   *  where it is rendered, so this stays a pure, React-free check. */
  reason: MessageDescriptor | null;
}

/** The subset of settings needed to judge styling-LLM readiness. */
export interface LlmSettings {
  llmEnabled: boolean;
  /** Remote provider config; the key is masked ('••••••••') when saved. */
  openaiLlmApiKey: string;
  openaiLlmModel: string;
}

const STYLING_OFF: LlmReadiness = {
  ready: false,
  reason: msg`Styling is turned off. Enable a styling engine in Settings before generating examples.`,
};

const PROVIDER_NOT_CONFIGURED: LlmReadiness = {
  ready: false,
  reason: msg`Your styling provider isn’t fully configured. Add its API key and model in Settings before generating examples.`,
};

/**
 * Whether the styling LLM is ready to (re)generate few-shot examples: styling
 * must be enabled and the remote provider's API key and model set. Pure and
 * synchronous — the remote provider needs no local model load, so this is a
 * plain configuration check with no IPC round-trip.
 */
export function getLlmReadiness(settings: LlmSettings): LlmReadiness {
  if (!settings.llmEnabled) return STYLING_OFF;
  if (settings.openaiLlmApiKey.trim() === '' || settings.openaiLlmModel.trim() === '') {
    return PROVIDER_NOT_CONFIGURED;
  }
  return { ready: true, reason: null };
}
