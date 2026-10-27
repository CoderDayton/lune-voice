// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, CircleAlert, Gauge, type LucideIcon, Mic } from 'lucide-react';
import { type ReactNode, useId } from 'react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import type {
  AsrBackendValue,
  ParakeetModelValue,
  WhisperModelValue,
} from '../../settings/data/settings-data';
import {
  dictationLanguagesFor,
  localizeOptions,
  localizeText,
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
  icon: LucideIcon;
}

/**
 * First wizard screen: the on-device speech engines, Whisper and Parakeet, and
 * the size of the one chosen. The cloud backend is intentionally omitted here
 * — onboarding is about getting a local model on disk; the remote option lives
 * in Settings.
 *
 * Both choices are hairline-divided rows in one panel, the same shape as the
 * Settings pages, with each control on the right of the row it belongs to.
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
  const { t, i18n } = useLingui();
  const engineGroup = useId();
  const sizeGroup = useId();

  const engines: EngineOption[] = [
    {
      value: 'WhisperCpp',
      label: t`Whisper`,
      tagline: t`Multilingual · runs on-device`,
      icon: Mic,
    },
    {
      value: 'ParakeetCpp',
      label: t`Parakeet`,
      tagline: t`Fastest on CPU · 25 European languages`,
      icon: Gauge,
    },
  ];
  const parakeet = engineBackend === 'ParakeetCpp';
  const sizes = parakeet ? parakeetModels : whisperModels;
  const selectedSize = parakeet ? engineParakeet : engineWhisper;

  return (
    <div className="flex flex-col gap-5">
      <fieldset>
        <legend className="setup-label">{t`Speech engine`}</legend>
        <div className="setup-group">
          {engines.map((engine) => {
            const Icon = engine.icon;
            const active = engineBackend === engine.value;
            return (
              <OptionRow
                key={engine.value}
                groupName={engineGroup}
                active={active}
                onSelect={() => onSelectEngine(engine.value)}
                leading={
                  <span
                    className={`setup-tile transition-colors ${
                      active ? 'bg-primary/15 text-primary' : 'bg-surface-wash text-content-subtle'
                    }`}
                  >
                    <Icon size={18} strokeWidth={1.8} />
                  </span>
                }
                title={engine.label}
                subtitle={engine.tagline}
              />
            );
          })}
        </div>
      </fieldset>

      <fieldset>
        <legend className="setup-label">{t`Model size`}</legend>
        <div className="setup-group">
          {sizes.map((m) => (
            <OptionRow
              key={m.value}
              groupName={sizeGroup}
              active={selectedSize === m.value}
              onSelect={() =>
                parakeet
                  ? onSelectParakeet(m.value as ParakeetModelValue)
                  : onSelectWhisper(m.value as WhisperModelValue)
              }
              compact
              title={localizeText(i18n, m.label)}
              subtitle={m.description === undefined ? undefined : localizeText(i18n, m.description)}
              trailing={<span className="text-xs tabular-nums text-content-subtle">{m.size}</span>}
            />
          ))}
          {parakeet && (
            <div className="setup-row-divider flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-3">
              <span className="min-w-0 flex-1 basis-60">
                <span className="block text-sm font-medium text-content-strong">
                  {t`Dictation language`}
                </span>
                <span className="mt-0.5 block text-xs leading-snug text-content-subtle">
                  {t`Parakeet cannot tell Lune which language it heard, so set the one you speak. You can change it later in Settings.`}
                </span>
              </span>
              <DescriptiveDropdown
                value={engineLanguage}
                options={localizeOptions(i18n, dictationLanguagesFor('ParakeetCpp'))}
                ariaLabel={t`Dictation language`}
                onChange={(next) => onSelectLanguage(next)}
                buttonClassName="sm:w-56"
              />
            </div>
          )}
        </div>
      </fieldset>

      {error && (
        <div className="alert alert-error gap-2 text-sm" role="alert">
          <CircleAlert size={18} className="shrink-0" />
          <span className="min-w-0 flex-1 break-words">{error}</span>
        </div>
      )}
    </div>
  );
}

interface OptionRowProps {
  groupName: string;
  active: boolean;
  onSelect: () => void;
  title: string;
  subtitle?: string;
  leading?: ReactNode;
  trailing?: ReactNode;
  /** One line per row, for a list the user scans rather than reads. */
  compact?: boolean;
}

/** One radio choice as a row of a panel: marker, text, an optional trailing value. */
function OptionRow({
  groupName,
  active,
  onSelect,
  title,
  subtitle,
  leading,
  trailing,
  compact = false,
}: OptionRowProps) {
  return (
    <label
      className={`setup-row-divider group flex cursor-pointer items-center gap-3 px-4 transition-colors -outline-offset-2 has-focus-visible:outline has-focus-visible:outline-2 has-focus-visible:outline-primary ${
        compact ? 'py-2' : 'py-3'
      } ${active ? 'bg-primary/8' : 'hover:bg-surface-hover'}`}
    >
      <input
        type="radio"
        name={groupName}
        className="sr-only"
        checked={active}
        onChange={onSelect}
      />
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
      {leading}
      <span className="min-w-0 flex-1">
        <span
          className={`block text-sm font-medium ${active ? 'text-content-strong' : 'text-content-default'}`}
        >
          {title}
        </span>
        {subtitle && (
          <span className="mt-0.5 block truncate text-xs text-content-subtle">{subtitle}</span>
        )}
      </span>
      {trailing}
    </label>
  );
}
