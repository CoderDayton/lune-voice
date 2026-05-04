export type ActivationModeValue = 'Toggle' | 'Hold';
export type AsrBackendValue = 'Qwen3Asr' | 'OpenAiCompatible';

export type DictationModeValue = 'TypeInPlace' | 'Clipboard' | 'Both';
export type PillVisibilityValue = 'Always' | 'WhenDictating' | 'Never';

export interface Option<Value extends string> {
  value: Value;
  label: string;
}

export const activationModes: Option<ActivationModeValue>[] = [
  { value: 'Toggle', label: 'Toggle to start and stop' },
  { value: 'Hold', label: 'Hold while speaking' },
];

export const pillVisibilityOptions: Option<PillVisibilityValue>[] = [
  { value: 'Always', label: 'Always' },
  { value: 'WhenDictating', label: 'When dictating' },
  { value: 'Never', label: 'Never' },
];

export const asrBackends: Option<AsrBackendValue>[] = [
  { value: 'Qwen3Asr', label: 'Qwen3-ASR 0.6B (local)' },
  { value: 'OpenAiCompatible', label: 'Cloud API (OpenAI-compatible)' },
];

export const deliveryMethods: Option<DictationModeValue>[] = [
  { value: 'Clipboard', label: 'Paste into app' },
  { value: 'TypeInPlace', label: 'Type keystrokes' },
  { value: 'Both', label: 'Paste and type' },
];

export type DictationLanguageValue =
  | 'auto'
  | 'en'
  | 'de'
  | 'es'
  | 'fr'
  | 'it'
  | 'ja'
  | 'ko'
  | 'pt'
  | 'ru'
  | 'zh';

export const dictationLanguages: Option<DictationLanguageValue>[] = [
  { value: 'auto', label: 'Auto-detect' },
  { value: 'en', label: 'English' },
  { value: 'de', label: 'Deutsch' },
  { value: 'es', label: 'Español' },
  { value: 'fr', label: 'Français' },
  { value: 'it', label: 'Italiano' },
  { value: 'ja', label: '日本語' },
  { value: 'ko', label: '한국어' },
  { value: 'pt', label: 'Português' },
  { value: 'ru', label: 'Русский' },
  { value: 'zh', label: '中文' },
];

export function labelFor<V extends string>(options: Option<V>[], value: V): string {
  return options.find((o) => o.value === value)?.label ?? value;
}
