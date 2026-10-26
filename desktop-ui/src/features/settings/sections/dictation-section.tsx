// SPDX-License-Identifier: GPL-3.0-or-later

import { Trans, useLingui } from '@lingui/react/macro';
import { invoke } from '@tauri-apps/api/core';
import { Keyboard, Mic } from 'lucide-react';
import { useEffect, useState } from 'react';
import {
  DescriptiveDropdown,
  type DescriptiveOption,
} from '../../../components/ui/descriptive-dropdown';
import { backendClient } from '../../../lib/backend/client';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { KeyCaptureInput } from '../components/key-capture-input';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { activationModes, agcModes, localizeOptions } from '../data/settings-data';

/** Empty string is the picker's sentinel for "use the system default input". */
const DEFAULT_DEVICE = '';

export function DictationSection({ settings }: SettingsProps) {
  const { t, i18n } = useLingui();

  const toggleCancelConflict =
    settings.hotkeyToggle.trim().toLowerCase() === settings.hotkeyCancel.trim().toLowerCase();

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

  // `null` until known, so the Hold warning never flashes on a working setup.
  // Re-read on every settings snapshot, not only a switch to Hold: any save
  // (a hotkey change, say) can start the native listener while Hold stays selected.
  const [holdAvailable, setHoldAvailable] = useState<boolean | null>(null);
  useEffect(() => {
    if (settings.activationMode !== 'Hold') return;
    backendClient
      .holdToTalkAvailable()
      .then(setHoldAvailable)
      .catch((e) => {
        console.error('hold_to_talk_available:', e);
        setHoldAvailable(null);
      });
  }, [settings]);
  const holdDegraded = settings.activationMode === 'Hold' && holdAvailable === false;

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
    <div className="flex flex-col gap-space-md">
      {/* Triggering — the hotkeys and how a press behaves. */}
      <SettingsSection
        title={t`Shortcuts`}
        description={t`Start, stop, and cancel dictation without leaving your work.`}
        icon={Keyboard}
      >
        <SettingsRow
          label={t`Start or stop recording`}
          description={t`The main shortcut used for voice typing. Click the binding to change it.`}
          control={
            <KeyCaptureInput
              value={settings.hotkeyToggle}
              onChange={(combo) => void saveWorkspaceSetting('hotkeyToggle', combo)}
            />
          }
        />
        <SettingsRow
          label={t`Cancel current recording`}
          description={t`Discard the current draft and return to idle. Click the binding to change it.`}
          control={
            <KeyCaptureInput
              value={settings.hotkeyCancel}
              onChange={(combo) => void saveWorkspaceSetting('hotkeyCancel', combo)}
            />
          }
        />
        <SettingsRow
          label={t`Recording mode`}
          description={t`Toggle is easiest for longer dictation. Hold is better for quick bursts.`}
          control={
            <DescriptiveDropdown
              value={settings.activationMode}
              options={localizeOptions(i18n, activationModes)}
              ariaLabel={t`Recording mode`}
              onChange={(next) =>
                void saveWorkspaceSetting('activationMode', next as typeof settings.activationMode)
              }
            />
          }
        />
      </SettingsSection>

      {toggleCancelConflict && (
        <div role="alert" className="alert alert-warning alert-soft sm:flex-row">
          <span>
            <Trans>
              Toggle and cancel are bound to the same key combo. One will override the other.
            </Trans>
          </span>
        </div>
      )}

      {holdDegraded && (
        <div role="alert" className="alert alert-warning alert-soft sm:flex-row">
          <span>
            <Trans>
              Hold is unavailable right now, so the shortcut toggles recording instead. Grant
              keyboard access to use Hold.
            </Trans>
          </span>
        </div>
      )}

      {/* Capture — the device and how recording detects speech. */}
      <SettingsSection
        title={t`Microphone`}
        description={t`Capture device and how recording detects when you speak.`}
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
          label={t`Adaptive gain control`}
          description={t`Evens out the microphone level before transcription. Helps quiet mics and noisy rooms.`}
          control={
            <DescriptiveDropdown
              value={settings.agcMode}
              options={localizeOptions(i18n, agcModes)}
              ariaLabel={t`Adaptive gain control`}
              onChange={(next) =>
                void saveWorkspaceSetting('agcMode', next as typeof settings.agcMode)
              }
            />
          }
        />
      </SettingsSection>
    </div>
  );
}
