import { ClipboardPaste, Cpu, Hand, Keyboard, PowerOff, Server, ToggleLeft } from 'lucide-react';
import type { ComponentType } from 'react';

export type ActivationModeValue = 'Toggle' | 'Hold';
export type AsrBackendValue = 'OpenAiCompatible' | 'WhisperCpp';

export type WhisperModelValue = 'Small' | 'Medium' | 'LargeV3Turbo';

export type DictationModeValue = 'TypeInPlace' | 'Clipboard';
export type PillVisibilityValue = 'Always' | 'WhenDictating' | 'Never';
export type PillAnchorValue =
  | 'TopLeft'
  | 'TopCenter'
  | 'TopRight'
  | 'BottomLeft'
  | 'BottomCenter'
  | 'BottomRight';
export type PillSizeValue = 'Compact' | 'Standard' | 'Large';
export type PillOrientationValue = 'Horizontal' | 'Vertical';
export type PillCornerValue = 'Pill' | 'Rounded' | 'Square';

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

export const pillAnchorOptions: Option<PillAnchorValue>[] = [
  { value: 'TopLeft', label: 'Top left' },
  { value: 'TopCenter', label: 'Top center' },
  { value: 'TopRight', label: 'Top right' },
  { value: 'BottomLeft', label: 'Bottom left' },
  { value: 'BottomCenter', label: 'Bottom center' },
  { value: 'BottomRight', label: 'Bottom right' },
];

export const pillSizeOptions: Option<PillSizeValue>[] = [
  { value: 'Compact', label: 'Compact' },
  { value: 'Standard', label: 'Standard' },
  { value: 'Large', label: 'Large' },
];

export const pillOrientationOptions: Option<PillOrientationValue>[] = [
  { value: 'Horizontal', label: 'Horizontal' },
  { value: 'Vertical', label: 'Vertical' },
];

export const pillCornerOptions: Option<PillCornerValue>[] = [
  { value: 'Pill', label: 'Fully round' },
  { value: 'Rounded', label: 'Rounded' },
  { value: 'Square', label: 'Square' },
];

export const asrBackends: DescriptiveOptionData<AsrBackendValue>[] = [
  {
    value: 'WhisperCpp',
    label: 'Whisper (local)',
    description: 'Runs whisper.cpp on-device. Choose a model size below.',
    icon: Cpu,
  },
  {
    value: 'OpenAiCompatible',
    // Not "Cloud": the endpoint is whatever the user points it at, and a
    // self-hosted server on localhost is a supported — and common — choice.
    label: 'OpenAI-compatible API',
    description:
      'Sends audio to an OpenAI-compatible endpoint you configure — a server on your own machine, or a hosted API.',
    icon: Server,
  },
];

/** The one enabled styling-LLM engine (Off is carried separately). */
export type LlmBackendValue = 'OpenAiCompatible';

/** Engine selector shown in the Styling-engine settings section. */
export type LlmEngineValue = 'Disabled' | 'OpenAiCompatible';

export const llmEngines: DescriptiveOptionData<LlmEngineValue>[] = [
  {
    value: 'Disabled',
    label: 'Off',
    description: 'No styling — transcripts are delivered exactly as transcribed.',
    icon: PowerOff,
  },
  {
    value: 'OpenAiCompatible',
    label: 'OpenAI-compatible API',
    description:
      'Sends transcripts to an OpenAI-compatible endpoint you configure — Ollama, llama-server or LM Studio locally, or a hosted API.',
    icon: Server,
  },
];

export interface WhisperModelOption extends DescriptiveOptionData<WhisperModelValue> {
  /** On-disk model directory — matches the backend ModelSpec `local_name`. */
  localName: string;
  /** Approximate download size, e.g. "264 MB". */
  size: string;
}

/** whisper.cpp GGML model options (quantized multilingual), smallest to largest. */
export const whisperModels: WhisperModelOption[] = [
  {
    value: 'Small',
    label: 'Small',
    description: 'Fastest, lowest accuracy.',
    localName: 'whisper-small-q5_1',
    size: '190 MB',
  },
  {
    value: 'Medium',
    label: 'Medium',
    description: 'Balanced accuracy and speed.',
    localName: 'whisper-medium-q5_0',
    size: '539 MB',
  },
  {
    value: 'LargeV3Turbo',
    label: 'Large v3 Turbo',
    description: 'Best accuracy; turbo decoder keeps it fast.',
    localName: 'whisper-large-v3-turbo-q8_0',
    size: '874 MB',
  },
];

/**
 * Approximate download sizes keyed by on-disk model directory (`local_name`),
 * shown in the Models library before a model is installed. Once present, the
 * backend reports the actual on-disk byte size instead.
 */
export const modelDownloadSizes: Record<string, string> = Object.fromEntries(
  whisperModels.map((m) => [m.localName, m.size]),
);

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

export type HistoryRetentionValue = 'off' | '7' | '30' | '90' | '365' | 'forever';

export const historyRetentionOptions: Option<HistoryRetentionValue>[] = [
  { value: 'off', label: "Don't save history" },
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
  { value: '365', label: '1 year' },
  { value: 'forever', label: 'Keep forever' },
];

/**
 * The dropdown value for a stored `sessionHistoryDays`.
 *
 * Three choices ride in one nullable number: `null` is off and `0` is the
 * "keep forever" sentinel (`LIMITS.historyDaysForever`) — neither is a window.
 */
export function historyRetentionValue(days: number | null): HistoryRetentionValue {
  if (days == null) return 'off';
  if (days === 0) return 'forever';
  return String(days) as HistoryRetentionValue;
}

/** Inverse of {@link historyRetentionValue}: the number to persist. */
export function historyRetentionDays(value: HistoryRetentionValue): number | null {
  if (value === 'off') return null;
  if (value === 'forever') return 0;
  return Number(value);
}

export type ModelIdleUnloadValue = 'never' | '5' | '15' | '30' | '60' | '120';

export const modelIdleUnloadOptions: DescriptiveOptionData<ModelIdleUnloadValue>[] = [
  {
    value: 'never',
    label: 'Keep loaded',
    description: 'The model stays in memory. Every dictation starts instantly.',
  },
  { value: '5', label: 'After 5 minutes' },
  { value: '15', label: 'After 15 minutes' },
  { value: '30', label: 'After 30 minutes' },
  { value: '60', label: 'After 1 hour' },
  { value: '120', label: 'After 2 hours' },
];

/** The dropdown value for a stored `modelIdleUnloadMinutes`; `null` is off. */
export function modelIdleUnloadValue(minutes: number | null): ModelIdleUnloadValue {
  if (minutes == null) return 'never';
  const exact = String(minutes) as ModelIdleUnloadValue;
  if (modelIdleUnloadOptions.some((o) => o.value === exact)) return exact;
  // A hand-edited settings.toml can hold a timeout that isn't on the list.
  // Snap it to the nearest listed one: the dropdown falls back to its *first*
  // option for an unknown value, which would read "Keep loaded" for a setting
  // that does in fact unload.
  const timed = modelIdleUnloadOptions
    .map((o) => o.value)
    .filter((v): v is Exclude<ModelIdleUnloadValue, 'never'> => v !== 'never');
  return timed.reduce((best, v) =>
    Math.abs(Number(v) - minutes) < Math.abs(Number(best) - minutes) ? v : best,
  );
}

/** Inverse of {@link modelIdleUnloadValue}: the number of minutes to persist. */
export function modelIdleUnloadMinutes(value: ModelIdleUnloadValue): number | null {
  return value === 'never' ? null : Number(value);
}

export function labelFor<V extends string>(options: Option<V>[], value: V): string {
  return options.find((o) => o.value === value)?.label ?? value;
}
