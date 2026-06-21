import { useLingui } from '@lingui/react/macro';
import { invoke } from '@tauri-apps/api/core';
import { Mic, Wand2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import {
  DescriptiveDropdown,
  type DescriptiveOption,
} from '../../../components/ui/descriptive-dropdown';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { CalibrateModal } from '../components/calibrate-modal';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { activationModes } from '../data/settings-data';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

/** Empty string is the picker's sentinel for "use the system default input". */
const DEFAULT_DEVICE = '';

export function InputSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const [startThreshold, setStartThreshold] = useDebouncedSetting(
    'speechThreshold',
    settings.speechThreshold,
  );
  const [endThreshold, setEndThreshold] = useDebouncedSetting(
    'speechThresholdEnd',
    settings.speechThresholdEnd,
  );

  const startNum = Number(startThreshold);
  const endNum = Number(endThreshold);
  const [calibrateOpen, setCalibrateOpen] = useState(false);

  // `null` until enumeration resolves, so we don't flag a valid saved device as
  // "unavailable" before the list has loaded.
  const [devices, setDevices] = useState<string[] | null>(null);
  useEffect(() => {
    invoke<string[]>('list_audio_input_devices')
      .then(setDevices)
      .catch((e) => {
        console.error('list_audio_input_devices:', e);
        setDevices([]);
      });
  }, []);

  const saved = settings.inputDevice;
  const available = devices ?? [];
  const deviceOptions: DescriptiveOption<string>[] = [
    { value: DEFAULT_DEVICE, label: t`System default`, icon: Mic },
    ...available.map((d) => ({ value: d, label: d })),
  ];
  // Keep the saved device selectable even when it isn't in the enumerated list:
  // while still loading, show it plainly; once loaded and still missing, mark it
  // unavailable (the backend falls back to the default to capture).
  if (saved && !available.includes(saved)) {
    deviceOptions.push({ value: saved, label: devices ? t`${saved} (unavailable)` : saved });
  }

  return (
    <SettingsSection
      title={t`Input`}
      description={t`How recording starts and stops, plus microphone tuning.`}
      icon={Mic}
    >
      <SettingsRow
        label={t`Microphone`}
        description={t`Which input device dictation captures from. Changes apply right away.`}
        control={
          <DescriptiveDropdown
            value={saved ?? DEFAULT_DEVICE}
            options={deviceOptions}
            ariaLabel={t`Microphone`}
            onChange={(next) =>
              void saveWorkspaceSetting('inputDevice', next === DEFAULT_DEVICE ? null : next)
            }
          />
        }
      />
      <SettingsRow
        label={t`Recording mode`}
        description={t`Toggle is easiest for longer dictation. Hold is better for quick bursts.`}
        control={
          <DescriptiveDropdown
            value={settings.activationMode}
            options={activationModes}
            ariaLabel={t`Recording mode`}
            onChange={(next) =>
              void saveWorkspaceSetting('activationMode', next as typeof settings.activationMode)
            }
          />
        }
      />

      <SettingsRow
        label={t`Microphone calibration`}
        description={t`Measure your room and voice to set both sensitivities automatically.`}
        control={
          <button
            type="button"
            className="btn btn-sm btn-soft gap-2"
            onClick={() => setCalibrateOpen(true)}
            aria-label={t`Calibrate microphone`}
          >
            <Wand2 size={14} strokeWidth={1.8} />
            {t`Calibrate`}
          </button>
        }
      />

      <SettingsRow
        label={t`Start sensitivity`}
        description={t`How easily Lune Voice triggers on speech. Higher = picks up quieter onsets.`}
        control={
          <div className="flex items-center gap-3">
            <input
              type="range"
              className="range range-primary range-sm w-28 sm:w-36"
              min={0.1}
              max={0.8}
              step={0.05}
              value={(1 - startNum).toFixed(2)}
              onChange={(e) => {
                const sensitivity = Number(e.target.value);
                const nextThreshold = (1 - sensitivity).toFixed(2);
                setStartThreshold(nextThreshold);
                if (Number(nextThreshold) < endNum) setEndThreshold(nextThreshold);
              }}
              aria-label={t`Start sensitivity`}
            />
            <span className="text-sm tabular-nums text-content-muted w-12 text-right">
              {(1 - startNum).toFixed(2)}
            </span>
          </div>
        }
      />

      <SettingsRow
        label={t`Pause sensitivity`}
        description={t`How quickly Lune Voice decides you're done speaking. Higher = ends sooner on a pause.`}
        control={
          <div className="flex items-center gap-3">
            <input
              type="range"
              className="range range-primary range-sm w-28 sm:w-36"
              min={0.1}
              max={0.9}
              step={0.05}
              value={endThreshold}
              onChange={(e) => {
                const next = e.target.value;
                const clamped = Number(next) > startNum ? String(startNum) : next;
                setEndThreshold(clamped);
              }}
              aria-label={t`Pause sensitivity`}
            />
            <span className="text-sm tabular-nums text-content-muted w-12 text-right">
              {endNum.toFixed(2)}
            </span>
          </div>
        }
      />

      <SettingsRow
        label={t`Adaptive gain control`}
        description={t`Normalizes microphone level before voice detection. Helps quiet mics and noisy rooms. Recommended.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.vadPcen}
            onChange={() => void saveWorkspaceSetting('vadPcen', !settings.vadPcen)}
            aria-label={t`Adaptive gain control`}
          />
        }
      />

      <SettingsRow
        label={t`Adaptive pause detection`}
        description={t`Learns your speaking rhythm and adjusts how long a pause ends a phrase. Helps if you pause mid-sentence to think.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.adaptiveVad}
            onChange={() => void saveWorkspaceSetting('adaptiveVad', !settings.adaptiveVad)}
            aria-label={t`Adaptive pause detection`}
          />
        }
      />

      <SettingsRow
        label={t`Pre-warm microphone`}
        description={t`Keep the mic open in the background so dictation starts ~70ms sooner. The OS may show a recording indicator while Lune Voice is running.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.micPrewarm}
            onChange={() => void saveWorkspaceSetting('micPrewarm', !settings.micPrewarm)}
            aria-label={t`Pre-warm microphone`}
          />
        }
      />
      <CalibrateModal
        open={calibrateOpen}
        onClose={() => setCalibrateOpen(false)}
        currentStart={startNum}
        currentEnd={endNum}
      />
    </SettingsSection>
  );
}
