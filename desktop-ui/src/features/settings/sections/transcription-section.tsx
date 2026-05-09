import { useLingui } from '@lingui/react/macro';
import { AudioWaveform } from 'lucide-react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { OpenAiFields } from '../components/openai-fields';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { asrBackends, dictationLanguages } from '../data/settings-data';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

export function TranscriptionSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const [confidence, setConfidence] = useDebouncedSetting(
    'confidenceThreshold',
    settings.confidenceThreshold,
  );

  return (
    <SettingsSection
      title="Recognition"
      description={t`Control how your speech is turned into text.`}
      icon={AudioWaveform}
    >
      <SettingsRow
        label={t`Voice model`}
        description={t`Qwen3-ASR runs locally with automatic language detection.`}
        control={
          <DescriptiveDropdown
            value={settings.asrBackend}
            options={asrBackends}
            ariaLabel={t`Voice model`}
            onChange={(next) =>
              void saveWorkspaceSetting('asrBackend', next as typeof settings.asrBackend)
            }
          />
        }
      />
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
      <SettingsRow
        label={t`Correction level`}
        description={t`How aggressively the LLM corrects transcribed words. Higher values send more words through correction.`}
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
              aria-label={t`Correction level`}
            />
            <span className="text-sm tabular-nums text-content-muted w-8 text-right">
              {Number(confidence) < 0.35 ? t`Low` : Number(confidence) > 0.7 ? t`High` : t`Mid`}
            </span>
          </div>
        }
      />
    </SettingsSection>
  );
}
