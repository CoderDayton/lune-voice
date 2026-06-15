import { useLingui } from '@lingui/react/macro';
import { SlidersHorizontal } from 'lucide-react';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

export function AdvancedSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const [confidence, setConfidence] = useDebouncedSetting(
    'confidenceThreshold',
    settings.confidenceThreshold,
  );
  const [llmSkipThreshold, setLlmSkipThreshold] = useDebouncedSetting(
    'llmSkipThreshold',
    settings.llmSkipThreshold,
  );
  const confidenceValue = Number(confidence);
  const threshold = Number(llmSkipThreshold);

  return (
    <SettingsSection
      title={t`Advanced`}
      description={t`Tune low-level recognition and correction behavior.`}
      icon={SlidersHorizontal}
    >
      <SettingsRow
        label={t`Correction level`}
        description={t`Sets the per-word confidence cutoff for correction. Higher values mark more words as uncertain; lower values trust more words as-is.`}
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
            <span className="text-sm tabular-nums text-content-muted w-12 text-right">
              {confidenceValue.toFixed(2)}
            </span>
          </div>
        }
      />
      <SettingsRow
        label={t`LLM sensitivity`}
        description={t`Sets the clean-segment bypass cutoff. Higher values send more otherwise-resolved segments to the LLM; lower values skip the LLM more often.`}
        control={
          <div className="flex items-center gap-3">
            <input
              type="range"
              className="range range-primary range-sm w-28 sm:w-36"
              min={0.1}
              max={0.95}
              step={0.05}
              value={threshold.toFixed(2)}
              onChange={(e) => setLlmSkipThreshold(e.target.value)}
              aria-label={t`LLM sensitivity`}
            />
            <span className="text-sm tabular-nums text-content-muted w-12 text-right">
              {threshold.toFixed(2)}
            </span>
          </div>
        }
      />
    </SettingsSection>
  );
}
