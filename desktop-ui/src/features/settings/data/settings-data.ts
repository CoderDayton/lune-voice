import {
  ClipboardCopy,
  ClipboardPaste,
  Cloud,
  Cpu,
  Hand,
  Keyboard,
  ToggleLeft,
} from 'lucide-react';
import type { ComponentType } from 'react';

export type ActivationModeValue = 'Toggle' | 'Hold';
export type AsrBackendValue = 'Qwen3Asr' | 'OpenAiCompatible';

export type DictationModeValue = 'TypeInPlace' | 'Clipboard' | 'Both';
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
    value: 'OpenAiCompatible',
    label: 'Cloud API (OpenAI-compatible)',
    description: 'Sends audio to a remote OpenAI-compatible endpoint you configure.',
    icon: Cloud,
  },
];

export const deliveryMethods: DescriptiveOptionData<DictationModeValue>[] = [
  {
    value: 'Clipboard',
    label: 'Paste into app',
    description: 'Sets the clipboard, then sends a paste shortcut. Fastest for long dictations.',
    icon: ClipboardPaste,
  },
  {
    value: 'TypeInPlace',
    label: 'Type into app',
    description: 'Simulates keystrokes one character at a time. Works where paste is blocked.',
    icon: Keyboard,
  },
  {
    value: 'Both',
    label: 'Type into app and copy',
    description: 'Types the text into the focused app and keeps it on the clipboard for re-use.',
    icon: ClipboardCopy,
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

export function labelFor<V extends string>(options: Option<V>[], value: V): string {
  return options.find((o) => o.value === value)?.label ?? value;
}
