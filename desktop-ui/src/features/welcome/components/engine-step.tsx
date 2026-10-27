// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, CircleAlert, Gauge, type LucideIcon, Mic } from 'lucide-react';
import { useId } from 'react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import type {
  AsrBackendValue,
  ParakeetModelValue,
  WhisperModelValue,
} from '../../settings/data/settings-data';
import {
  dictationLanguagesFor,
  formatLanguageList,
  localizeOptions,
  localizeText,
  parakeetLanguages,
  parakeetModels,
  whisperModels,
} from '../../settings/data/settings-data';

interface EngineStepProps {
  engineBackend: AsrBackendValue;
  engineWhisper: WhisperModelValue;
  engineParakeet: ParakeetModelValue;
  /** Dictation language; with Parakeet always one it covers. */
  engineLanguage: string;
  error: string | null;
  onSelectEngine: (backend: AsrBackendValue) => void;
  onSelectWhisper: (model: WhisperModelValue) => void;
  onSelectParakeet: (model: ParakeetModelValue) => void;
  onSelectLanguage: (language: string) => void;
}

interface EngineOption {
  value: Extract<AsrBackendValue, 'WhisperCpp' | 'ParakeetCpp'>;
  label: string;
  tagline: string;
  detail: string;
  icon: LucideIcon;
}

/**
 * Both option grids size their columns by what fits, not by a fixed count, so
 * another engine or model size lands as one more tile and the screen wraps
 * instead of growing a scrollbar.
 */
const ENGINE_GRID = 'grid gap-3 grid-cols-[repeat(auto-fit,minmax(16rem,1fr))]';
const SIZE_GRID = 'grid gap-3 grid-cols-[repeat(auto-fit,minmax(11rem,1fr))]';

/**
 * First wizard screen: the on-device speech engines, Whisper and Parakeet, and
 * the size of the one chosen. The cloud backend is intentionally omitted here
 * — onboarding is about getting a local model on disk; the remote option lives
 * in Settings.
 */
export function EngineStep({
  engineBackend,
  engineWhisper,
  engineParakeet,
  engineLanguage,
  error,
  onSelectEngine,
  onSelectWhisper,
  onSelectParakeet,
  onSelectLanguage,
}: EngineStepProps) {
  const { t } = useLingui();

  const engines: EngineOption[] = [
    {
      value: 'WhisperCpp',
      label: t`Whisper`,
      tagline: t`Multilingual · runs on-device`,
      detail: t`whisper.cpp running on-device. Choose a size to trade speed for accuracy.`,
      icon: Mic,
    },
    {
      value: 'ParakeetCpp',
      label: t`Parakeet`,
      tagline: t`Fastest on CPU · 25 European languages`,
      detail: t`NVIDIA Parakeet running on-device. Punctuates by itself and detects which of its languages you speak.`,
      icon: Gauge,
    },
  ];

  return (
    <div className="flex flex-col gap-5">
      {/* A picker needs something to pick between. With one engine the grid
          left a half-width card marooned beside empty space and asked the
          user to select the only option there was, so a single engine is
          stated instead — same card, no false choice. The grid returns on
          its own the moment a second engine does. */}
      {engines.length > 1 ? (
        <section>
          <span className="setup-label">{t`Speech engine`}</span>
          <div className={ENGINE_GRID}>
            {engines.map((engine) => (
              <EngineCard
                key={engine.value}
                option={engine}
                active={engineBackend === engine.value}
                onSelect={() => onSelectEngine(engine.value)}
              />
            ))}
          </div>
        </section>
      ) : (
        engines[0] && <EngineSummary option={engines[0]} />
      )}

      {engineBackend === 'WhisperCpp' && (
        <WhisperSizeStep selected={engineWhisper} onSelect={onSelectWhisper} />
      )}
      {engineBackend === 'ParakeetCpp' && (
        <ParakeetSizeStep
          selected={engineParakeet}
          onSelect={onSelectParakeet}
          language={engineLanguage}
          onSelectLanguage={onSelectLanguage}
        />
      )}

      {error && (
        <div className="alert alert-error gap-2 text-sm" role="alert">
          <CircleAlert size={18} className="shrink-0" />
          <span className="min-w-0 flex-1 break-words">{error}</span>
        </div>
      )}
    </div>
  );
}

interface WhisperSizeStepProps {
  selected: WhisperModelValue;
  onSelect: (model: WhisperModelValue) => void;
}

/** The Whisper size choice, one tile per size. */
function WhisperSizeStep({ selected, onSelect }: WhisperSizeStepProps) {
  const { t, i18n } = useLingui();
  const groupName = useId();

  return (
    <fieldset>
      <legend className="setup-label">{t`Model size`}</legend>
      <div className={SIZE_GRID}>
        {whisperModels.map((m) => (
          <SizeTile
            key={m.value}
            label={localizeText(i18n, m.label)}
            description={m.description === undefined ? '' : localizeText(i18n, m.description)}
            size={m.size}
            groupName={groupName}
            active={selected === m.value}
            onSelect={() => onSelect(m.value)}
          />
        ))}
      </div>
    </fieldset>
  );
}

interface SizeTileProps {
  label: string;
  description: string;
  size: string;
  groupName: string;
  active: boolean;
  onSelect: () => void;
}

function SizeTile({ label, description, size, groupName, active, onSelect }: SizeTileProps) {
  return (
    <label
      className={`group flex cursor-pointer flex-col gap-1.5 rounded-box border p-3 transition-colors has-focus-visible:outline has-focus-visible:outline-2 has-focus-visible:outline-primary ${
        active
          ? 'border-primary/50 bg-primary/8'
          : 'border-surface-border bg-surface-panel hover:border-primary/45 hover:bg-surface-hover'
      }`}
    >
      <input
        type="radio"
        name={groupName}
        className="sr-only"
        checked={active}
        onChange={onSelect}
      />
      {/* Every option carries a marker, selected or not, so the tiles read as
          a set of choices. The size sits on the right of the same row: a
          column of sizes is the number the user is actually weighing. */}
      <span className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2">
          <span
            className={`setup-radio mt-0 ${
              active
                ? 'border-primary bg-primary'
                : 'border-surface-border-strong group-hover:border-primary/60'
            }`}
            aria-hidden="true"
          >
            {active && <Check size={11} strokeWidth={3.5} className="text-primary-content" />}
          </span>
          <span
            className={`text-sm font-medium ${
              active ? 'text-content-strong' : 'text-content-default'
            }`}
          >
            {label}
          </span>
        </span>
        <span className="shrink-0 text-xs tabular-nums text-content-subtle">{size}</span>
      </span>
      <span className="text-xs leading-snug text-content-subtle">{description}</span>
    </label>
  );
}

interface ParakeetSizeStepProps {
  selected: ParakeetModelValue;
  onSelect: (model: ParakeetModelValue) => void;
  language: string;
  onSelectLanguage: (language: string) => void;
}

/**
 * The Parakeet size choice, in the same tiles as Whisper's, plus the language
 * list — the limit someone weighing Parakeet against Whisper needs to see
 * before committing to a download — and the dictation language, which
 * Parakeet needs set because it cannot report the one it heard.
 */
function ParakeetSizeStep({
  selected,
  onSelect,
  language,
  onSelectLanguage,
}: ParakeetSizeStepProps) {
  const { t, i18n } = useLingui();
  const groupName = useId();
  const languages = formatLanguageList(i18n.locale, parakeetLanguages);

  return (
    <fieldset>
      <legend className="setup-label">{t`Model size`}</legend>
      <div className={SIZE_GRID}>
        {parakeetModels.map((m) => (
          <SizeTile
            key={m.value}
            label={localizeText(i18n, m.label)}
            description={m.description === undefined ? '' : localizeText(i18n, m.description)}
            size={m.size}
            groupName={groupName}
            active={selected === m.value}
            onSelect={() => onSelect(m.value)}
          />
        ))}
      </div>
      <p className="mt-2 text-xs leading-snug text-content-subtle">
        {t`Covers ${languages}. For any other language, choose Whisper.`}
      </p>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-content-strong">{t`Dictation language`}</span>
          <span className="mt-0.5 block text-xs leading-snug text-content-subtle">
            {t`Parakeet cannot tell Lune which language it heard, so set the one you speak. You can change it later in Settings.`}
          </span>
        </span>
        <DescriptiveDropdown
          value={language}
          options={localizeOptions(i18n, dictationLanguagesFor('ParakeetCpp'))}
          ariaLabel={t`Dictation language`}
          onChange={(next) => onSelectLanguage(next)}
        />
      </div>
    </fieldset>
  );
}

/**
 * The chosen engine, stated rather than offered, for when there is only one.
 */
function EngineSummary({ option }: { option: EngineOption }) {
  const Icon = option.icon;
  return (
    <div className="setup-card flex items-start gap-3 p-4">
      <span className="setup-tile bg-primary/15 text-primary">
        <Icon size={18} strokeWidth={1.8} />
      </span>
      <div className="min-w-0 flex-1">
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="text-sm font-medium text-content-strong">{option.label}</span>
          <span className="text-xs text-content-subtle">{option.tagline}</span>
        </p>
        <p className="mt-1 text-xs leading-snug text-content-subtle">{option.detail}</p>
      </div>
    </div>
  );
}

function EngineCard({
  option,
  active,
  onSelect,
}: {
  option: EngineOption;
  active: boolean;
  onSelect: () => void;
}) {
  const Icon = option.icon;
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onSelect}
      className={`group flex items-start gap-3 rounded-box border p-4 text-start transition-all duration-150 ${
        active
          ? 'border-primary/50 bg-primary/8 shadow-sm'
          : 'border-surface-border bg-surface-panel hover:border-primary/45 hover:bg-primary/8'
      }`}
    >
      <span
        className={`setup-tile transition-colors ${
          active
            ? 'bg-primary/15 text-primary'
            : 'bg-surface-wash text-content-subtle group-hover:text-primary'
        }`}
      >
        <Icon size={18} strokeWidth={1.8} />
      </span>
      <span className="min-w-0 flex-1">
        <span
          className={`text-sm font-medium ${active ? 'text-content-strong' : 'text-content-default'}`}
        >
          {option.label}
        </span>
        <span className="mt-0.5 block text-xs text-content-subtle">{option.tagline}</span>
        <span className="mt-1.5 block text-xs leading-snug text-content-subtle">
          {option.detail}
        </span>
      </span>
      <span
        className={`mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border-2 transition-colors ${
          active ? 'border-primary bg-primary' : 'border-surface-border-strong'
        }`}
        aria-hidden="true"
      >
        {active && <Check size={12} strokeWidth={3} className="text-primary-content" />}
      </span>
    </button>
  );
}
