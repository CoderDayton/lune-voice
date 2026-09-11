import type { I18n, MessageDescriptor } from '@lingui/core';
import { msg } from '@lingui/core/macro';
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

/**
 * Option text as stored here: a message to translate at render time, or a
 * plain string for text that must not be translated — endonyms in the spoken
 * language list, product names, sizes.
 */
export type OptionText = string | MessageDescriptor;

export interface Option<Value extends string> {
  value: Value;
  label: OptionText;
}

type IconComponent = ComponentType<{
  size?: number;
  strokeWidth?: number;
  className?: string;
}>;

export interface DescriptiveOptionData<Value extends string> {
  value: Value;
  label: OptionText;
  description?: OptionText;
  icon?: IconComponent;
}

/** Resolved option text, ready to render. */
export interface LocalizedOption<Value extends string> {
  value: Value;
  label: string;
  description?: string;
  icon?: IconComponent;
}

export function localizeText(i18n: I18n, text: OptionText): string {
  return typeof text === 'string' ? text : i18n._(text);
}

/**
 * Translate an option list for the active locale. Call it at render time —
 * the module-level lists hold descriptors, not text, so a locale switch has to
 * re-resolve them.
 */
export function localizeOptions<Value extends string>(
  i18n: I18n,
  options: readonly DescriptiveOptionData<Value>[],
): LocalizedOption<Value>[] {
  return options.map((option) => ({
    value: option.value,
    icon: option.icon,
    label: localizeText(i18n, option.label),
    description:
      option.description === undefined ? undefined : localizeText(i18n, option.description),
  }));
}

export const activationModes: DescriptiveOptionData<ActivationModeValue>[] = [
  {
    value: 'Toggle',
    label: msg`Toggle to start and stop`,
    description: msg`Press the hotkey to start, press again to stop. Easiest for longer dictations.`,
    icon: ToggleLeft,
  },
  {
    value: 'Hold',
    label: msg`Hold while speaking`,
    description: msg`Hold the hotkey while speaking, release to stop. Better for quick bursts.`,
    icon: Hand,
  },
];

export const pillVisibilityOptions: Option<PillVisibilityValue>[] = [
  { value: 'Always', label: msg`Always` },
  { value: 'WhenDictating', label: msg`When dictating` },
  { value: 'Never', label: msg`Never` },
];

export const pillAnchorOptions: Option<PillAnchorValue>[] = [
  { value: 'TopLeft', label: msg`Top left` },
  { value: 'TopCenter', label: msg`Top center` },
  { value: 'TopRight', label: msg`Top right` },
  { value: 'BottomLeft', label: msg`Bottom left` },
  { value: 'BottomCenter', label: msg`Bottom center` },
  { value: 'BottomRight', label: msg`Bottom right` },
];

export const pillSizeOptions: Option<PillSizeValue>[] = [
  { value: 'Compact', label: msg`Compact` },
  { value: 'Standard', label: msg`Standard` },
  { value: 'Large', label: msg`Large` },
];

export const pillOrientationOptions: Option<PillOrientationValue>[] = [
  { value: 'Horizontal', label: msg`Horizontal` },
  { value: 'Vertical', label: msg`Vertical` },
];

export const pillCornerOptions: Option<PillCornerValue>[] = [
  { value: 'Pill', label: msg`Fully round` },
  { value: 'Rounded', label: msg`Rounded` },
  { value: 'Square', label: msg`Square` },
];

export const asrBackends: DescriptiveOptionData<AsrBackendValue>[] = [
  {
    value: 'WhisperCpp',
    label: msg`Whisper (local)`,
    description: msg`Runs whisper.cpp on-device. Choose a model size below.`,
    icon: Cpu,
  },
  {
    value: 'OpenAiCompatible',
    // Not "Cloud": the endpoint is whatever the user points it at, and a
    // self-hosted server on localhost is a supported — and common — choice.
    label: msg`OpenAI-compatible API`,
    description: msg`Sends audio to an OpenAI-compatible endpoint you configure — a server on your own machine, or a hosted API.`,
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
    label: msg`Off`,
    description: msg`No styling — transcripts are delivered exactly as transcribed.`,
    icon: PowerOff,
  },
  {
    value: 'OpenAiCompatible',
    label: msg`OpenAI-compatible API`,
    description: msg`Sends transcripts to an OpenAI-compatible endpoint you configure — Ollama, llama-server or LM Studio locally, or a hosted API.`,
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
    label: msg`Small`,
    description: msg`Fastest, lowest accuracy.`,
    localName: 'whisper-small-q5_1',
    size: '190 MB',
  },
  {
    value: 'Medium',
    label: msg`Medium`,
    description: msg`Balanced accuracy and speed.`,
    localName: 'whisper-medium-q5_0',
    size: '539 MB',
  },
  {
    value: 'LargeV3Turbo',
    label: msg`Large v3 Turbo`,
    description: msg`Best accuracy; turbo decoder keeps it fast.`,
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
    label: msg`Paste into app`,
    description: msg`Sends a paste shortcut into the focused app. Fastest for long dictations.`,
    icon: ClipboardPaste,
  },
  {
    value: 'TypeInPlace',
    label: msg`Type into app`,
    description: msg`Simulates keystrokes one character at a time. Works where paste is blocked.`,
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
  {
    value: 'auto',
    label: msg`Auto-detect`,
    description: msg`Detects the spoken language each session.`,
  },
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
    label: msg`Automatic`,
    description: msg`Detects terminals and picks the matching paste shortcut. Works on most compositors.`,
  },
  {
    value: 'Normal',
    label: msg`Always Ctrl+V`,
    description: msg`Force the standard paste shortcut for every app.`,
  },
  {
    value: 'Terminal',
    label: msg`Always Ctrl+Shift+V`,
    description: msg`Force the terminal paste shortcut. Use on COSMIC if you mostly dictate into terminals.`,
  },
];

export type PrivacyModeValue = 'LocalOnly' | 'LocalPreferred' | 'AllowRemote';

export const privacyModes: DescriptiveOptionData<PrivacyModeValue>[] = [
  {
    value: 'LocalOnly',
    label: msg`Local only`,
    description: msg`Never send audio or text to a remote service. Remote transcription and styling are disabled.`,
  },
  {
    value: 'LocalPreferred',
    label: msg`Local preferred`,
    description: msg`Prefer on-device processing, but allow a remote service where you have configured one.`,
  },
  {
    value: 'AllowRemote',
    label: msg`Allow remote`,
    description: msg`Allow remote transcription and styling wherever you have configured them.`,
  },
];

export type HistoryRetentionValue = 'off' | '7' | '30' | '90' | '365' | 'forever';

export const historyRetentionOptions: Option<HistoryRetentionValue>[] = [
  { value: 'off', label: msg`Don't save history` },
  { value: '7', label: msg`7 days` },
  { value: '30', label: msg`30 days` },
  { value: '90', label: msg`90 days` },
  { value: '365', label: msg`1 year` },
  { value: 'forever', label: msg`Keep forever` },
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
    label: msg`Keep loaded`,
    description: msg`The model stays in memory. Every dictation starts instantly.`,
  },
  { value: '5', label: msg`After 5 minutes` },
  { value: '15', label: msg`After 15 minutes` },
  { value: '30', label: msg`After 30 minutes` },
  { value: '60', label: msg`After 1 hour` },
  { value: '120', label: msg`After 2 hours` },
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

export function labelFor<V extends string>(i18n: I18n, options: Option<V>[], value: V): string {
  const label = options.find((o) => o.value === value)?.label;
  return label === undefined ? value : localizeText(i18n, label);
}

/**
 * Compact labels for the Overview status line, where the full option label is
 * too long. Separate messages rather than trimming the long ones: the suffixes
 * that used to be stripped ("… into app") only exist in English.
 */
export const asrBackendShortLabels: Record<AsrBackendValue, MessageDescriptor> = {
  WhisperCpp: msg`Whisper`,
  OpenAiCompatible: msg`OpenAI-compatible`,
};

export const dictationModeShortLabels: Record<DictationModeValue, MessageDescriptor> = {
  Clipboard: msg`Paste`,
  TypeInPlace: msg`Type`,
};
