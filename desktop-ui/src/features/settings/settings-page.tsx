// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import {
  AudioLines,
  type LucideIcon,
  Mic,
  Palette,
  Pill,
  Send,
  SlidersHorizontal,
  Wand2,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { NAV_ITEM_BASE, NavBead, navIconClass, navItemState } from '../../components/ui/nav-item';
import { workspaceStore } from '../../store/workspace-store';
import type { SettingsSectionId } from './sections';
import { AppearanceSection } from './sections/appearance-section';
import { DictationSection } from './sections/dictation-section';
import { GeneralSection } from './sections/general-section';
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
  ];

  const content: Record<SectionId, ReactNode> = {
    general: <GeneralSection settings={settings} />,
    appearance: <AppearanceSection />,
    dictation: <DictationSection settings={settings} />,
    pill: <PillSection />,
    transcription: <TranscriptionSection settings={settings} />,
    styling: <StylingEngineSection settings={settings} />,
    output: <OutputSection settings={settings} />,
  };

  return (
    // Laid out on the page's own width (a container query), not the window's:
    // the app sidebar takes a share of the window, so a window breakpoint gave
    // the rail a quarter of an already narrow page and its labels spilled out.
    <div className="@container">
      <div className="grid gap-space-md @3xl:grid-cols-[13.5rem_minmax(0,1fr)]">
        {/* Section rail — a scrolling tab strip when the page is narrow, a
          vertical rail of fixed width once there is room for one. Only the
          rail stays pinned: a pinned strip left a band above it where the
          page scrolled into view. */}
        <nav
          aria-label={t`Settings sections`}
          className="bento-cell z-20 flex-row gap-1 overflow-x-auto p-space-sm @3xl:sticky @3xl:top-0 @3xl:flex-col @3xl:self-start @3xl:overflow-x-visible"
        >
          {sections.map(({ id, label, icon: Icon }) => {
            const isActive = id === active;
            return (
              <button
                key={id}
                type="button"
                onClick={() => setActive(id)}
                aria-current={isActive ? 'page' : undefined}
                title={label}
                className={`${NAV_ITEM_BASE} h-12 shrink-0 pr-4 pl-2 text-left @3xl:w-full @3xl:min-w-0 ${navItemState(isActive)}`}
              >
                <NavBead active={isActive}>
                  <Icon
                    size={17}
                    strokeWidth={isActive ? 2 : 1.7}
                    className={navIconClass(isActive)}
                  />
                </NavBead>
                {/* Whole in the tab strip; in the rail, a translation longer
                  than the rail ends in an ellipsis (full text in the title)
                  instead of running past the panel edge. */}
                <span className="whitespace-nowrap @3xl:min-w-0 @3xl:truncate">{label}</span>
              </button>
            );
          })}
        </nav>

        {/* All sections stay mounted; inactive ones are hidden so transient
          state (e.g. an in-progress model download) survives tab switches. */}
        <div className="min-w-0">
          {sections.map(({ id }) => (
            <div key={id} hidden={id !== active}>
              {content[id]}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
