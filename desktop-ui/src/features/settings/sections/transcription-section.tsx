import { useLingui } from '@lingui/react/macro';
import { AudioLines, SpellCheck } from 'lucide-react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { AsrEngineCards } from '../components/asr-engine-cards';
import { OpenAiFields } from '../components/openai-fields';
import {
  SettingsBlock,
  type SettingsProps,
  SettingsRow,
  SettingsSection,
} from '../components/section';
import { dictationLanguages } from '../data/settings-data';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

export function TranscriptionSection({
  settings,
  onManageModels,
}: SettingsProps & { onManageModels?: () => void }) {
  const { t } = useLingui();
  const [confidence, setConfidence] = useDebouncedSetting(
    'confidenceThreshold',
    settings.confidenceThreshold,
  );

  // Confidence-gated correction needs per-word probabilities, which only the
  // local recognizer provides — the remote (OpenAI-compatible) backend reports
  // none, so the sensitivity control has no effect there and the section hides.
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
          description={t`Choose the engine that turns your speech into text. Local engines run fully on-device; Whisper is multilingual with selectable model sizes.`}
        >
          <AsrEngineCards settings={settings} onManageModels={onManageModels} />
        </SettingsBlock>
        {settings.asrBackend === 'OpenAiCompatible' && (
          <OpenAiFields
            settings={settings}
            baseUrlKey="openaiAsrBaseUrl"
            apiKeyKey="openaiAsrApiKey"
            modelKey="openaiAsrModel"
            apiDescription="transcription API"
            modelDescription="Transcription model name, e.g. whisper-1."
            modelPlaceholder="whisper-1"
          />
        )}
        <SettingsRow
          label={t`Dictation language`}
          description={t`Set the language you speak. Auto uses detection from speech.`}
          control={
            <DescriptiveDropdown
              value={settings.dictationLanguage}
              options={dictationLanguages}
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
      </SettingsSection>

      {/* Words → corrected words — an on-device heuristic corrector. It matches
          against your dictionary and known vocabulary and re-cases known terms;
          per-word confidence decides which uncertain words to check. No AI
          model and no rewriting — it never invents text, only fixes it. */}
      {hasWordConfidence && (
        <SettingsSection
          title={t`Correction`}
          description={t`Fix recognizer errors with your dictionary and known vocabulary, on your device. It uses per-word confidence and statistical patterns — no AI model, and it never rewrites what you said.`}
          icon={SpellCheck}
        >
          <SettingsRow
            label={t`Correction sensitivity`}
            description={t`The words the recognizer is least sure about are checked against your dictionary and vocabulary. Higher checks more words; lower trusts the raw transcription.`}
            control={
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  className="range range-primary range-sm w-28 sm:w-36"
                  min={0.1}
                  max={0.95}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => setConfidence(e.target.value)}
                  aria-label={t`Correction sensitivity`}
                />
                <span className="text-sm tabular-nums text-content-muted w-12 text-right">
                  {Number(confidence).toFixed(2)}
                </span>
              </div>
            }
          />
        </SettingsSection>
      )}
    </div>
  );
}
