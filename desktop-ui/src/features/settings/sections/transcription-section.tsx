// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { AudioLines, SpellCheck } from 'lucide-react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { LIMITS } from '../../../lib/backend/generated/limits';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { AsrEngineCards } from '../components/asr-engine-cards';
import { GpuBackendCard } from '../components/gpu-backend-card';
import {
  SettingsBlock,
  type SettingsProps,
  SettingsRow,
  SettingsSection,
} from '../components/section';
import {
  dictationLanguagesFor,
  localizeOptions,
  modelIdleUnloadMinutes,
  modelIdleUnloadOptions,
  modelIdleUnloadValue,
} from '../data/settings-data';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

export function TranscriptionSection({ settings }: SettingsProps) {
  const { t, i18n } = useLingui();
  const [confidence, setConfidence] = useDebouncedSetting(
    'confidenceThreshold',
    settings.confidenceThreshold,
  );

  // Confidence-gated correction needs per-word probabilities, which only the
  // local recognizers (Whisper, Parakeet) provide — the remote
  // (OpenAI-compatible) backend reports none, so the sensitivity control has
  // no effect there and the section hides.
  const hasWordConfidence = settings.asrBackend !== 'OpenAiCompatible';

  return (
    <div className="flex flex-col gap-space-md">
      {/* Speech → text — the recognizer that turns audio into words. */}
      <SettingsSection
        title={t`Transcription`}
        description={t`Control how your speech is turned into text.`}
        icon={AudioLines}
      >
        <SettingsBlock
          label={t`Voice model`}
          description={t`What turns your speech into text. The local engines run on your machine.`}
        >
          <AsrEngineCards settings={settings} />
        </SettingsBlock>
        {settings.asrBackend === 'OpenAiCompatible' && settings.privacyMode === 'LocalOnly' && (
          <div role="alert" className="alert alert-warning alert-soft sm:flex-row">
            <span>{t`Remote services are turned off, so dictation cannot start with this engine. Turn on Allow remote services in General, or switch to an on-device engine.`}</span>
          </div>
        )}
        <SettingsRow
          label={t`Dictation language`}
          description={
            settings.asrBackend === 'ParakeetCpp'
              ? t`The language you speak. Parakeet needs it set.`
              : t`Set the language you speak. Auto uses detection from speech.`
          }
          control={
            <DescriptiveDropdown
              value={settings.dictationLanguage}
              options={localizeOptions(
                i18n,
                dictationLanguagesFor(settings.asrBackend, settings.dictationLanguage),
              )}
              ariaLabel={t`Dictation language`}
              onChange={(next) =>
                void saveWorkspaceSetting(
                  'dictationLanguage',
                  next as typeof settings.dictationLanguage,
                )
              }
            />
          }
        />
        <SettingsRow
          label={t`Use hardware acceleration`}
          description={t`When available, this can make transcription finish faster.`}
          control={
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={settings.useGpu}
              onChange={() => void saveWorkspaceSetting('useGpu', !settings.useGpu)}
              aria-label={t`Use hardware acceleration`}
            />
          }
        />
        {settings.useGpu && <GpuBackendCard />}
        <SettingsRow
          label={t`Free the model when idle`}
          description={t`Give the memory back after a quiet spell. The next dictation reloads the model, which adds up to a second before it starts listening — your speech is recorded during the wait, so nothing is lost.`}
          control={
            <DescriptiveDropdown
              value={modelIdleUnloadValue(settings.modelIdleUnloadMinutes)}
              options={localizeOptions(i18n, modelIdleUnloadOptions)}
              ariaLabel={t`Free the model when idle`}
              onChange={(next) =>
                void saveWorkspaceSetting('modelIdleUnloadMinutes', modelIdleUnloadMinutes(next))
              }
            />
          }
        />
      </SettingsSection>

      {/* Words → corrected words — an on-device heuristic corrector. It matches
          against your dictionary and known vocabulary and re-cases known terms;
          per-word confidence decides which uncertain words to check. No AI
          model and no rewriting — it never invents text, only fixes it. */}
      <SettingsSection
        title={t`Correction`}
        description={t`Fix recognizer errors with your dictionary and known vocabulary, on your device. It uses per-word confidence and statistical patterns — no AI model, and it never rewrites what you said.`}
        icon={SpellCheck}
      >
        <SettingsRow
          label={t`Capitalize known names`}
          description={t`Write names like GitHub and iPhone the way their owners do. Turn this off to keep the recognizer's own capitals. A style can override this.`}
          control={
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={settings.brandCasing}
              onChange={() => void saveWorkspaceSetting('brandCasing', !settings.brandCasing)}
              aria-label={t`Capitalize known names`}
            />
          }
        />
        {hasWordConfidence && (
          <SettingsRow
            label={t`Correction sensitivity`}
            description={t`The words the recognizer is least sure about are checked against your dictionary and vocabulary. Higher checks more words; lower trusts the raw transcription.`}
            control={
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  className="range range-primary range-sm w-28 sm:w-36"
                  // A deliberately narrower range than the backend accepts
                  // (`LIMITS.confidenceMin`/`Max`): the extremes are valid but
                  // useless — 0 corrects nothing, 1 second-guesses every word.
                  min={Math.max(LIMITS.confidenceMin, 0.1)}
                  max={Math.min(LIMITS.confidenceMax, 0.95)}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => setConfidence(e.target.value)}
                  aria-label={t`Correction sensitivity`}
                />
                <span className="text-sm tabular-nums text-content-muted w-12 text-end">
                  {Number(confidence).toFixed(2)}
                </span>
              </div>
            }
          />
        )}
      </SettingsSection>
    </div>
  );
}
