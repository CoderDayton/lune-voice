import { ClipboardPaste, Cloud, Cpu, Hand, Keyboard, ToggleLeft } from 'lucide-react';
import type { ComponentType } from 'react';

export type ActivationModeValue = 'Toggle' | 'Hold';
export type AsrBackendValue = 'Qwen3Asr' | 'OpenAiCompatible' | 'WhisperCpp';

export type WhisperModelValue = 'Small' | 'Medium' | 'LargeV3Turbo';

export type DictationModeValue = 'TypeInPlace' | 'Clipboard';
export type PillVisibilityValue = 'Always' | 'WhenDictating' | 'Never';

export interface Option<Value extends string> {
  value: Value;
  label: string;
}

type IconComponent = ComponentType<{
  size?: number;
  strokeWidth?: number;
  className?: string;
}>;

export interface DescriptiveOptionData<Value extends string> {
  value: Value;
  label: string;
  description?: string;
  icon?: IconComponent;
}

export const activationModes: DescriptiveOptionData<ActivationModeValue>[] = [
  {
    value: 'Toggle',
    label: 'Toggle to start and stop',
    description: 'Press the hotkey to start, press again to stop. Easiest for longer dictations.',
    icon: ToggleLeft,
  },
  {
    value: 'Hold',
    label: 'Hold while speaking',
    description: 'Hold the hotkey while speaking, release to stop. Better for quick bursts.',
    icon: Hand,
  },
];

export const pillVisibilityOptions: Option<PillVisibilityValue>[] = [
  { value: 'Always', label: 'Always' },
  { value: 'WhenDictating', label: 'When dictating' },
  { value: 'Never', label: 'Never' },
];

export const asrBackends: DescriptiveOptionData<AsrBackendValue>[] = [
  {
    value: 'Qwen3Asr',
    label: 'Qwen3-ASR 0.6B (local)',
    description: 'Runs on-device. Audio never leaves your machine.',
    icon: Cpu,
  },
  {
    value: 'WhisperCpp',
    label: 'Whisper (local)',
    description: 'Runs whisper.cpp on-device. Choose a model size below.',
    icon: Cpu,
  },
  {
    value: 'OpenAiCompatible',
    label: 'Cloud API (OpenAI-compatible)',
    description: 'Sends audio to a remote OpenAI-compatible endpoint you configure.',
    icon: Cloud,
  },
];

/** whisper.cpp GGML model options (q8_0 multilingual), smallest to largest. */
export const whisperModels: DescriptiveOptionData<WhisperModelValue>[] = [
  {
    value: 'Small',
    label: 'Small',
    description: '~264 MB. Fastest, lowest accuracy.',
  },
  {
    value: 'Medium',
    label: 'Medium',
    description: '~823 MB. Balanced accuracy and speed.',
  },
  {
    value: 'LargeV3Turbo',
    label: 'Large v3 Turbo',
    description: '~874 MB. Best accuracy; turbo decoder keeps it fast.',
  },
];

export const deliveryMethods: DescriptiveOptionData<DictationModeValue>[] = [
  {
    value: 'Clipboard',
    label: 'Paste into app',
    description: 'Sends a paste shortcut into the focused app. Fastest for long dictations.',
    icon: ClipboardPaste,
  },
  {
    value: 'TypeInPlace',
    label: 'Type into app',
    description: 'Simulates keystrokes one character at a time. Works where paste is blocked.',
    icon: Keyboard,
  },
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

export const dictationLanguages: DescriptiveOptionData<DictationLanguageValue>[] = [
  { value: 'auto', label: 'Auto-detect', description: 'Detects the spoken language each session.' },
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

export type TerminalModeValue = 'Auto' | 'Terminal' | 'Normal';

export const terminalModes: DescriptiveOptionData<TerminalModeValue>[] = [
  {
    value: 'Auto',
    label: 'Automatic',
    description:
      'Detects terminals and picks the matching paste shortcut. Works on most compositors.',
  },
  {
    value: 'Normal',
    label: 'Always Ctrl+V',
    description: 'Force the standard paste shortcut for every app.',
  },
  {
    value: 'Terminal',
    label: 'Always Ctrl+Shift+V',
    description:
      'Force the terminal paste shortcut. Use on COSMIC if you mostly dictate into terminals.',
  },
];

export type HistoryRetentionValue = 'off' | '7' | '30' | '90' | '365';

export const historyRetentionOptions: Option<HistoryRetentionValue>[] = [
  { value: 'off', label: "Don't save history" },
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
  { value: '365', label: '1 year' },
];

export function labelFor<V extends string>(options: Option<V>[], value: V): string {
  return options.find((o) => o.value === value)?.label ?? value;
}
