import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Move, Pill, Shapes } from 'lucide-react';
import { useEffect, useState } from 'react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { backendClient } from '../../../lib/backend/client';
import type { MonitorInfoDto } from '../../../lib/backend/types';
import { saveWorkspaceSetting, workspaceStore } from '../../../store/workspace-store';
import { SettingsRow, SettingsSection } from '../components/section';
import {
  pillAnchorOptions,
  pillCornerOptions,
  pillOrientationOptions,
  pillSizeOptions,
  pillVisibilityOptions,
} from '../data/settings-data';
import { useDebouncedSetting } from '../hooks/use-debounced-setting';

/** Sentinel for "no pinned monitor" in the dropdown (a real connector name is
 *  never empty). */
const AUTO_MONITOR = '';

/**
 * Settings for the floating dictation pill — the small status overlay shown
 * while dictating. Visibility and behaviour, placement, and size & shape.
 */
export function PillSection() {
  const { t } = useLingui();
  const settings = useStore(workspaceStore, (s) => s.settings);
  const [monitors, setMonitors] = useState<MonitorInfoDto[]>([]);
  const [edgeMargin, setEdgeMargin] = useDebouncedSetting(
    'pillEdgeMargin',
    settings.pillEdgeMargin,
  );

  useEffect(() => {
    let cancelled = false;
    backendClient.listMonitors().then(
      (list) => {
        if (!cancelled) setMonitors(list);
      },
      (e) => console.warn('[settings] monitor list failed', e),
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const monitorOptions = [
    { value: AUTO_MONITOR, label: t`Automatic` },
    ...monitors.map((m) => ({
      value: m.name,
      label: `${m.name} — ${m.width}×${m.height}${m.isPrimary ? ` · ${t`primary`}` : ''}`,
    })),
  ];
  // Keep a saved-but-disconnected monitor selectable so it isn't silently
  // dropped just because the screen is unplugged right now.
  if (settings.pillMonitor && !monitors.some((m) => m.name === settings.pillMonitor)) {
    monitorOptions.push({
      value: settings.pillMonitor,
      label: `${settings.pillMonitor} — ${t`not connected`}`,
    });
  }

  return (
    <div className="page-stack">
      <SettingsSection
        title={t`Dictation pill`}
        description={t`The floating status pill you see while dictating.`}
        icon={Pill}
      >
        <SettingsRow
          label={t`Visibility`}
          description={t`When the pill appears while you dictate.`}
          control={
            <DescriptiveDropdown
              value={settings.pillVisibility}
              options={pillVisibilityOptions}
              ariaLabel={t`Dictation pill visibility`}
              onChange={(next) =>
                void saveWorkspaceSetting('pillVisibility', next as typeof settings.pillVisibility)
              }
            />
          }
        />
        <SettingsRow
          label={t`Show on all workspaces`}
          description={t`Keep the pill with you when you switch workspaces or virtual desktops.`}
          control={
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={settings.pillAllWorkspaces}
              onChange={() =>
                void saveWorkspaceSetting('pillAllWorkspaces', !settings.pillAllWorkspaces)
              }
              aria-label={t`Show on all workspaces`}
            />
          }
        />
      </SettingsSection>

      <SettingsSection
        title={t`Placement`}
        description={t`Where the pill sits on screen. Dragging it overrides the preset until you reset.`}
        icon={Move}
      >
        <SettingsRow
          label={t`Position`}
          description={t`Preset corner or edge the pill starts at.`}
          control={
            <DescriptiveDropdown
              value={settings.pillAnchor}
              options={pillAnchorOptions}
              ariaLabel={t`Pill position preset`}
              onChange={(next) =>
                void saveWorkspaceSetting('pillAnchor', next as typeof settings.pillAnchor)
              }
            />
          }
        />
        <SettingsRow
          label={t`Edge distance`}
          description={t`Gap between the pill and the screen edge, so it clears a taskbar or dock.`}
          control={
            <label className="input input-sm w-28">
              <input
                type="number"
                min={0}
                max={512}
                step={4}
                value={edgeMargin}
                onChange={(e) => {
                  const raw = Number(e.target.value);
                  setEdgeMargin(
                    Number.isFinite(raw) ? Math.max(0, Math.min(512, Math.round(raw))) : 0,
                  );
                }}
                aria-label={t`Edge distance in pixels`}
              />
              <span className="label">px</span>
            </label>
          }
        />
        <SettingsRow
          label={t`Monitor`}
          description={t`Which screen the pill appears on.`}
          control={
            <DescriptiveDropdown
              value={settings.pillMonitor ?? AUTO_MONITOR}
              options={monitorOptions}
              ariaLabel={t`Pill monitor`}
              onChange={(next) =>
                void saveWorkspaceSetting('pillMonitor', next === AUTO_MONITOR ? null : next)
              }
            />
          }
        />
        <SettingsRow
          label={t`Lock position`}
          description={t`Prevent moving the pill by dragging it.`}
          control={
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={settings.pillLocked}
              onChange={() => void saveWorkspaceSetting('pillLocked', !settings.pillLocked)}
              aria-label={t`Lock position`}
            />
          }
        />
        <SettingsRow
          label={t`Reset position`}
          description={t`Forget where you dragged the pill and return it to its preset spot.`}
          control={
            <button
              type="button"
              className="btn btn-sm"
              onClick={() =>
                void backendClient
                  .resetPillPosition()
                  .catch((e) => console.error('[settings] reset pill position failed', e))
              }
            >
              {t`Reset`}
            </button>
          }
        />
      </SettingsSection>

      <SettingsSection
        title={t`Size & shape`}
        description={t`How the pill bar looks. Changes apply immediately.`}
        icon={Shapes}
      >
        <SettingsRow
          label={t`Size`}
          description={t`Overall footprint of the pill bar.`}
          control={
            <DescriptiveDropdown
              value={settings.pillSize}
              options={pillSizeOptions}
              ariaLabel={t`Pill size`}
              onChange={(next) =>
                void saveWorkspaceSetting('pillSize', next as typeof settings.pillSize)
              }
            />
          }
        />
        <SettingsRow
          label={t`Orientation`}
          description={t`A vertical pill hugs the side of the screen.`}
          control={
            <DescriptiveDropdown
              value={settings.pillOrientation}
              options={pillOrientationOptions}
              ariaLabel={t`Pill orientation`}
              onChange={(next) =>
                void saveWorkspaceSetting(
                  'pillOrientation',
                  next as typeof settings.pillOrientation,
                )
              }
            />
          }
        />
        <SettingsRow
          label={t`Corners`}
          description={t`Corner rounding of the pill bar.`}
          control={
            <DescriptiveDropdown
              value={settings.pillCorner}
              options={pillCornerOptions}
              ariaLabel={t`Pill corners`}
              onChange={(next) =>
                void saveWorkspaceSetting('pillCorner', next as typeof settings.pillCorner)
              }
            />
          }
        />
      </SettingsSection>
    </div>
  );
}
