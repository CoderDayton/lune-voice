import { useLingui } from '@lingui/react/macro';
import { Mic } from 'lucide-react';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { activationModes } from '../data/settings-data';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

export function InputSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const [threshold, setThreshold] = useDebouncedSetting(
    'speechThreshold',
    settings.speechThreshold,
  );

  return (
    <SettingsSection
      title={t`Voice input`}
      description={t`Choose how quickly you want to enter and leave voice typing.`}
      icon={Mic}
    >
      <SettingsRow
        label={t`Recording mode`}
        description={t`Toggle is easiest for longer dictation. Hold is better for quick bursts.`}
        control={
          <select
            className="select w-full sm:w-auto sm:min-w-56"
            value={settings.activationMode}
            onChange={(event) =>
              void saveWorkspaceSetting(
                'activationMode',
                event.target.value as typeof settings.activationMode,
              )
            }
          >
            {activationModes.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        }
      />

      <SettingsRow
        label={t`Pause sensitivity`}
        description={t`How quickly Lune decides you're done speaking.`}
        control={
          <div className="flex items-center gap-3">
            <input
              type="range"
              className="range range-primary range-sm w-28 sm:w-36"
              min={0.2}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              aria-label={t`Pause sensitivity`}
            />
            <span className="text-sm tabular-nums text-content-muted w-8 text-right">
              {Number(threshold) < 0.35 ? t`Slow` : Number(threshold) > 0.65 ? t`Fast` : t`Mid`}
            </span>
          </div>
        }
      />
    </SettingsSection>
  );
}
