import { useLingui } from '@lingui/react/macro';
import { AudioLines, Wand2 } from 'lucide-react';
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

export function TranscriptionSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const llmEnabled = settings.llmEnabled;
  const [confidence, setConfidence] = useDebouncedSetting(
    'confidenceThreshold',
    settings.confidenceThreshold,
  );
  const [llmSkip, setLlmSkip] = useDebouncedSetting('llmSkipThreshold', settings.llmSkipThreshold);

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
          <AsrEngineCards settings={settings} />
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

      {/* Text → polished text — the local LLM that cleans up transcripts. */}
      <SettingsSection
        title={t`Correction`}
        description={t`Tune how the local LLM polishes transcripts after transcription.`}
        icon={Wand2}
      >
        <SettingsRow
          label={t`Correction LLM`}
          description={
            llmEnabled
              ? t`Polishes punctuation and fixes obvious recognizer errors after transcription.`
              : t`Disabled — transcripts are delivered raw, and the LLM model is hidden from the Models tab.`
          }
          control={
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={llmEnabled}
              onChange={(e) => void saveWorkspaceSetting('llmEnabled', e.target.checked)}
              aria-label={t`Enable correction LLM`}
            />
          }
        />

        {llmEnabled && (
          <>
            <SettingsRow
              label={t`Correction sensitivity`}
              description={t`Words the recognizer is unsure about are sent to the LLM. Higher corrects more aggressively; lower trusts the raw transcription.`}
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
            <SettingsRow
              label={t`Correction frequency`}
              description={t`How readily the LLM runs on clean speech. Higher polishes more segments; lower skips more often for speed.`}
              control={
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    className="range range-primary range-sm w-28 sm:w-36"
                    min={0.1}
                    max={0.95}
                    step={0.05}
                    value={Number(llmSkip).toFixed(2)}
                    onChange={(e) => setLlmSkip(e.target.value)}
                    aria-label={t`Correction frequency`}
                  />
                  <span className="text-sm tabular-nums text-content-muted w-12 text-right">
                    {Number(llmSkip).toFixed(2)}
                  </span>
                </div>
              }
            />
          </>
        )}
      </SettingsSection>
    </div>
  );
}
