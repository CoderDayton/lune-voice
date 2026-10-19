// SPDX-License-Identifier: GPL-3.0-or-later

import type { I18n, MessageDescriptor } from '@lingui/core';
import { msg } from '@lingui/core/macro';
import {
  ClipboardPaste,
  Cpu,
  Gauge,
  Hand,
  Keyboard,
  PowerOff,
  Server,
  ToggleLeft,
} from 'lucide-react';
import type { ComponentType } from 'react';

export type ActivationModeValue = 'Toggle' | 'Hold';
export type AsrBackendValue = 'OpenAiCompatible' | 'WhisperCpp' | 'ParakeetCpp';

export type WhisperModelValue = 'Base' | 'Small' | 'Medium' | 'LargeV3Turbo';

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
    value: 'ParakeetCpp',
    label: msg`Parakeet (local)`,
    description: msg`NVIDIA Parakeet on-device. Much faster than Whisper on a CPU, for 25 European languages.`,
    icon: Gauge,
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
    value: 'Base',
    label: msg`Base`,
    description: msg`Fastest, lowest accuracy. For older or low-power computers.`,
    localName: 'whisper-base-q8_0',
    size: '78 MB',
  },
  {
    value: 'Small',
    label: msg`Small`,
    description: msg`Fast; less accurate than Medium.`,
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

/** Parakeet quantizations — mirrors `ParakeetModel` in lune-core. */
export type ParakeetModelValue = 'TdtV3Q8' | 'TdtV3Q4K';

export interface ParakeetModelOption extends DescriptiveOptionData<ParakeetModelValue> {
  /** On-disk model directory — matches the backend ModelSpec `local_name`. */
  localName: string;
  /** Approximate download size. */
  size: string;
}

/**
 * NVIDIA parakeet-tdt-0.6b-v3, smallest to largest. Same weights, same
 * languages; the quantization trades a little accuracy for size.
 */
export const parakeetModels: ParakeetModelOption[] = [
  {
    value: 'TdtV3Q4K',
    label: msg`Compact`,
    description: msg`Smaller download and less memory, slightly less accurate.`,
    localName: 'parakeet-tdt-0.6b-v3-q4_k',
    size: '416 MB',
  },
  {
    value: 'TdtV3Q8',
    label: msg`Full`,
    description: msg`Practically the accuracy of the original model. Recommended.`,
    localName: 'parakeet-tdt-0.6b-v3-q8_0',
    size: '669 MB',
  },
];

/**
 * The languages Parakeet v3 transcribes, as ISO 639-1 codes. Mirrors
 * `ParakeetModel::languages` in lune-core, which refuses to start a dictation
 * in any other language.
 */
export const parakeetLanguages: readonly string[] = [
  'bg',
  'cs',
  'da',
  'de',
  'el',
  'en',
  'es',
  'et',
  'fi',
  'fr',
  'hr',
  'hu',
  'it',
  'lt',
  'lv',
  'mt',
  'nl',
  'pl',
  'pt',
  'ro',
  'ru',
  'sk',
  'sl',
  'sv',
  'uk',
];

/**
 * Whether Parakeet can transcribe a dictation-language setting. `auto` always
 * can: the model detects the language itself. Only the primary subtag counts,
 * so `pt-BR` is Portuguese.
 */
export function parakeetSupportsLanguage(tag: string): boolean {
  const trimmed = tag.trim();
  if (trimmed === '' || trimmed.toLowerCase() === 'auto') return true;
  const primary = trimmed.split(/[-_]/)[0]?.toLowerCase() ?? '';
  return parakeetLanguages.includes(primary);
}

/**
 * A list of languages, given as ISO 639-1 codes, written out in `locale`:
 * "Bulgarian, Croatian, … and Ukrainian" for English, "bulgare, croate, … et
 * ukrainien" for French, sorted the way that locale sorts. The names come from
 * the runtime's own CLDR data, so they need no translation of their own. Falls
 * back to the bare codes if the runtime cannot name languages.
 */
export function formatLanguageList(locale: string, codes: readonly string[]): string {
  try {
    const names = new Intl.DisplayNames([locale], { type: 'language' });
    const named = codes.map((code) => names.of(code) ?? code);
    named.sort((a, b) => a.localeCompare(b, locale));
    return new Intl.ListFormat([locale], { style: 'long', type: 'conjunction' }).format(named);
  } catch {
    return codes.join(', ');
  }
}

/**
 * Whether an engine can be set to a dictation language. Parakeet needs one
 * it covers, given explicitly: it recognizes its languages by itself but never
 * reports which one it heard, so with Auto every dictation would be formatted
 * by English rules. Mirrors `ParakeetModel::accepts_dictation_language`, which
 * the backend enforces on save. Whisper and the remote API take anything.
 */
export function engineAcceptsLanguage(backend: AsrBackendValue, tag: string): boolean {
  if (backend !== 'ParakeetCpp') return true;
  const trimmed = tag.trim();
  return trimmed !== '' && trimmed.toLowerCase() !== 'auto' && parakeetSupportsLanguage(trimmed);
}

/**
 * The dictation-language options to offer while `backend` is selected. When
 * the saved language is one the engine cannot use — only a hand-edited
 * settings file gets there — it stays listed first, so the selector shows
 * what is actually set instead of silently displaying another language.
 */
export function dictationLanguagesFor(
  backend: AsrBackendValue,
  current?: string,
): DescriptiveOptionData<DictationLanguageValue>[] {
  const offered = dictationLanguages.filter((o) => engineAcceptsLanguage(backend, o.value));
  if (current === undefined || offered.some((o) => o.value === current)) return offered;
  const stray = dictationLanguages.find((o) => o.value === current);
  return stray ? [stray, ...offered] : offered;
}

/**
 * The dictation language to use once `backend` is selected: the current one
 * when the engine takes it, otherwise the first of `preferred` it does (the
 * user's system languages, most wanted first), otherwise English. Only ever
 * moves off a language the engine cannot use.
 */
export function languageForEngine(
  backend: AsrBackendValue,
  current: string,
  preferred: readonly string[],
): string {
  if (engineAcceptsLanguage(backend, current)) return current;
  for (const tag of preferred) {
    const primary = tag.trim().split(/[-_]/)[0]?.toLowerCase() ?? '';
    if (
      engineAcceptsLanguage(backend, primary) &&
      dictationLanguages.some((o) => o.value === primary)
    ) {
      return primary;
    }
  }
  return 'en';
}

export type ModelChoice =
  | { engine: 'WhisperCpp'; model: WhisperModelValue }
  | { engine: 'ParakeetCpp'; model: ParakeetModelValue }
  | { engine: 'OpenAiCompatible' };

export interface ModelChoiceSettings {
  asrBackend: AsrBackendValue;
  dictationLanguage: string;
  whisperModel?: WhisperModelValue;
  parakeetModel?: ParakeetModelValue;
}

export function modelChoiceSettings(
  choice: ModelChoice,
  currentLanguage: string,
  preferred: readonly string[],
): ModelChoiceSettings {
  const dictationLanguage = languageForEngine(choice.engine, currentLanguage, preferred);
  if (choice.engine === 'WhisperCpp') {
    return { asrBackend: choice.engine, whisperModel: choice.model, dictationLanguage };
  }
  if (choice.engine === 'ParakeetCpp') {
    return { asrBackend: choice.engine, parakeetModel: choice.model, dictationLanguage };
  }
  return { asrBackend: choice.engine, dictationLanguage };
}

/** The system's preferred languages, most wanted first; empty off a browser. */
export function systemLanguages(): string[] {
  if (typeof navigator === 'undefined') return [];
  const list = navigator.languages?.length ? navigator.languages : [navigator.language];
  return list.filter((tag): tag is string => typeof tag === 'string' && tag !== '');
}

/** A dictation language's own name ("Deutsch"), or the tag if it has none. */
export function dictationLanguageName(i18n: I18n, tag: string): string {
  const option = dictationLanguages.find((o) => o.value === tag);
  return option ? localizeText(i18n, option.label) : tag;
}

/** The local model directory an engine choice depends on, if it has one. */
export function engineModelLocalName(
  backend: AsrBackendValue,
  whisper: WhisperModelValue,
  parakeet: ParakeetModelValue,
): string | null {
  switch (backend) {
    case 'WhisperCpp':
      return whisperModels.find((m) => m.value === whisper)?.localName ?? null;
    case 'ParakeetCpp':
      return parakeetModels.find((m) => m.value === parakeet)?.localName ?? null;
    case 'OpenAiCompatible':
      return null;
  }
}

/**
 * Approximate download sizes keyed by on-disk model directory (`local_name`),
 * shown in the Models library before a model is installed. Once present, the
 * backend reports the actual on-disk byte size instead.
 */
export const modelDownloadSizes: Record<string, string> = Object.fromEntries([
  ...whisperModels.map((m) => [m.localName, m.size]),
  ...parakeetModels.map((m) => [m.localName, m.size]),
]);

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
  | 'bg'
  | 'cs'
  | 'da'
  | 'de'
  | 'el'
  | 'es'
  | 'et'
  | 'fi'
  | 'fr'
  | 'hr'
  | 'hu'
  | 'it'
  | 'ja'
  | 'ko'
  | 'lt'
  | 'lv'
  | 'mt'
  | 'nl'
  | 'pl'
  | 'pt'
  | 'ro'
  | 'ru'
  | 'sk'
  | 'sl'
  | 'sv'
  | 'uk'
  | 'zh';

/**
 * Every dictation language the app offers: Auto-detect, English, then the
 * rest by code, each named in itself. Whisper takes all of them; Parakeet the
 * subset in {@link parakeetLanguages}, and never Auto — see
 * {@link dictationLanguagesFor}.
 */
export const dictationLanguages: DescriptiveOptionData<DictationLanguageValue>[] = [
  {
    value: 'auto',
    label: msg`Auto-detect`,
    description: msg`Detects the spoken language each session.`,
  },
  { value: 'en', label: 'English' },
  { value: 'bg', label: 'Български' },
  { value: 'cs', label: 'Čeština' },
  { value: 'da', label: 'Dansk' },
  { value: 'de', label: 'Deutsch' },
  { value: 'el', label: 'Ελληνικά' },
  { value: 'es', label: 'Español' },
  { value: 'et', label: 'Eesti' },
  { value: 'fi', label: 'Suomi' },
  { value: 'fr', label: 'Français' },
  { value: 'hr', label: 'Hrvatski' },
  { value: 'hu', label: 'Magyar' },
  { value: 'it', label: 'Italiano' },
  { value: 'ja', label: '日本語' },
  { value: 'ko', label: '한국어' },
  { value: 'lt', label: 'Lietuvių' },
  { value: 'lv', label: 'Latviešu' },
  { value: 'mt', label: 'Malti' },
  { value: 'nl', label: 'Nederlands' },
  { value: 'pl', label: 'Polski' },
  { value: 'pt', label: 'Português' },
  { value: 'ro', label: 'Română' },
  { value: 'ru', label: 'Русский' },
  { value: 'sk', label: 'Slovenčina' },
  { value: 'sl', label: 'Slovenščina' },
  { value: 'sv', label: 'Svenska' },
  { value: 'uk', label: 'Українська' },
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

export type PrivacyModeValue = 'LocalOnly' | 'AllowRemote';

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
  ParakeetCpp: msg`Parakeet`,
  OpenAiCompatible: msg`OpenAI-compatible`,
};

export const dictationModeShortLabels: Record<DictationModeValue, MessageDescriptor> = {
  Clipboard: msg`Paste`,
  TypeInPlace: msg`Type`,
};
