// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import {
  AudioLines,
  HardDrive,
  type LucideIcon,
  Mic,
  Palette,
  Pill,
  Send,
  SlidersHorizontal,
  Wand2,
} from 'lucide-react';
import type { ReactNode } from 'react';
import {
  NAV_ITEM_BASE,
  NavActiveBar,
  navIconClass,
  navItemState,
} from '../../components/ui/nav-item';
import { workspaceStore } from '../../store/workspace-store';
import type { SettingsSectionId } from './sections';
import { AppearanceSection } from './sections/appearance-section';
import { DictationSection } from './sections/dictation-section';
import { GeneralSection } from './sections/general-section';
import { ModelsSection } from './sections/models-section';
import { OutputSection } from './sections/output-section';
import { PillSection } from './sections/pill-section';
import { StylingEngineSection } from './sections/styling-engine-section';
import { TranscriptionSection } from './sections/transcription-section';

type SectionId = SettingsSectionId;

export function SettingsPage() {
  const { t } = useLingui();
  const settings = useStore(workspaceStore, (s) => s.settings);
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);
  const { tab } = useSearch({ from: '/settings' });
  const navigate = useNavigate({ from: '/settings' });

  // `?tab=` is the single source of truth. Mirroring it into local state would
  // desync the moment the user clicks a tab: a repeat deep-link (the sidebar's
  // More menu while Settings is already open) then navigates to a search
  // value the URL already holds, and nothing re-renders.
  const active: SectionId = tab ?? 'general';
  const setActive = (id: SectionId) => {
    void navigate({ search: { tab: id }, replace: true });
  };

  if (!loaded) {
    return (
      <div className="page-stack">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="panel-surface p-5 sm:p-6">
            <div className="skeleton-shimmer h-5 w-32 mb-3" />
            <div className="skeleton-shimmer h-3 w-48 mb-4" />
            <div className="space-y-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:justify-between sm:items-center">
                <div className="skeleton-shimmer h-4 w-24" />
                <div className="skeleton-shimmer h-8 w-36" />
              </div>
              <div className="flex flex-col gap-2 sm:flex-row sm:justify-between sm:items-center">
                <div className="skeleton-shimmer h-4 w-32" />
                <div className="skeleton-shimmer h-8 w-28" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const sections: { id: SectionId; label: string; icon: LucideIcon }[] = [
    { id: 'general', label: t`General`, icon: SlidersHorizontal },
    { id: 'appearance', label: t`Appearance`, icon: Palette },
    { id: 'dictation', label: t`Dictation`, icon: Mic },
    { id: 'pill', label: t`Pill`, icon: Pill },
    { id: 'transcription', label: t`Transcription`, icon: AudioLines },
    { id: 'styling', label: t`Styling`, icon: Wand2 },
    { id: 'output', label: t`Output`, icon: Send },
    { id: 'models', label: t`Models`, icon: HardDrive },
  ];

  const content: Record<SectionId, ReactNode> = {
    general: <GeneralSection settings={settings} />,
    appearance: <AppearanceSection />,
    dictation: <DictationSection settings={settings} />,
    pill: <PillSection />,
    transcription: (
      <TranscriptionSection settings={settings} onManageModels={() => setActive('models')} />
    ),
    styling: <StylingEngineSection settings={settings} />,
    output: <OutputSection settings={settings} />,
    models: <ModelsSection />,
  };

  return (
    <div className="bento-board">
      {/* Section rail — horizontal scroll on mobile, vertical on desktop */}
      <nav
        aria-label={t`Settings sections`}
        className="bento-cell sticky top-0 z-20 col-span-12 flex-row gap-1 overflow-x-auto p-space-sm md:col-span-3 md:flex-col md:self-start"
      >
        {sections.map(({ id, label, icon: Icon }) => {
          const isActive = id === active;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setActive(id)}
              aria-current={isActive ? 'page' : undefined}
              className={`${NAV_ITEM_BASE} shrink-0 py-2.5 pr-3.5 pl-3 text-left md:w-full ${navItemState(isActive)}`}
            >
              {isActive && (
                <>
                  {/* Active accent: a left bar on the vertical desktop rail, a
                      bottom bar on the horizontal mobile row. */}
                  <NavActiveBar className="hidden md:block" />
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-3 bottom-0 h-[3px] rounded-full bg-primary md:hidden"
                  />
                </>
              )}
              <Icon
                size={18}
                strokeWidth={isActive ? 2.1 : 1.7}
                className={navIconClass(isActive)}
              />
              {label}
            </button>
          );
        })}
      </nav>

      {/* All sections stay mounted; inactive ones are hidden so transient
          state (e.g. an in-progress model download) survives tab switches. */}
      <div className="col-span-12 min-w-0 md:col-span-9">
        {sections.map(({ id }) => (
          <div key={id} hidden={id !== active}>
            {content[id]}
          </div>
        ))}
      </div>
    </div>
  );
}
