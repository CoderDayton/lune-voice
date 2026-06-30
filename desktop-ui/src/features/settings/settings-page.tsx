import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import {
  AudioLines,
  HardDrive,
  type LucideIcon,
  Mic,
  Palette,
  Send,
  SlidersHorizontal,
} from 'lucide-react';
import { useState } from 'react';
import { workspaceStore } from '../../store/workspace-store';
import { AppearanceSection } from './sections/appearance-section';
import { DictationSection } from './sections/dictation-section';
import { GeneralSection } from './sections/general-section';
import { ModelsSection } from './sections/models-section';
import { OutputSection } from './sections/output-section';
import { TranscriptionSection } from './sections/transcription-section';

type SectionId = 'general' | 'appearance' | 'dictation' | 'transcription' | 'output' | 'models';

export function SettingsPage() {
  const { t } = useLingui();
  const settings = useStore(workspaceStore, (s) => s.settings);
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);
  const [active, setActive] = useState<SectionId>('general');

  if (!loaded) {
    return (
      <div className="page-stack">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="card border rounded-box p-6 border-surface-border bg-surface-card"
          >
            <div className="skeleton h-5 w-32 mb-3" />
            <div className="skeleton h-3 w-48 mb-4" />
            <div className="space-y-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:justify-between sm:items-center">
                <div className="skeleton h-4 w-24" />
                <div className="skeleton h-8 w-36" />
              </div>
              <div className="flex flex-col gap-2 sm:flex-row sm:justify-between sm:items-center">
                <div className="skeleton h-4 w-32" />
                <div className="skeleton h-8 w-28" />
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
    { id: 'transcription', label: t`Transcription`, icon: AudioLines },
    { id: 'output', label: t`Output`, icon: Send },
    { id: 'models', label: t`Models`, icon: HardDrive },
  ];

  const content: Record<SectionId, React.ReactNode> = {
    general: <GeneralSection settings={settings} />,
    appearance: <AppearanceSection />,
    dictation: <DictationSection settings={settings} />,
    transcription: (
      <TranscriptionSection settings={settings} onManageModels={() => setActive('models')} />
    ),
    output: <OutputSection settings={settings} />,
    models: <ModelsSection />,
  };

  return (
    <div className="bento-board">
      {/* Section rail — horizontal scroll on mobile, vertical on desktop */}
      <nav
        aria-label={t`Settings sections`}
        className="bento-cell col-span-12 flex-row gap-1 overflow-x-auto p-space-sm md:col-span-3 md:flex-col md:self-start"
      >
        {sections.map(({ id, label, icon: Icon }) => {
          const isActive = id === active;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setActive(id)}
              aria-current={isActive ? 'page' : undefined}
              className={`flex shrink-0 items-center gap-3 rounded-field px-3.5 py-3 text-left text-[0.95rem] transition-colors focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-primary md:w-full ${
                isActive
                  ? 'bg-surface-nav-active font-medium text-content-strong'
                  : 'text-content-muted hover:bg-surface-hover hover:text-content'
              }`}
            >
              <Icon
                size={19}
                strokeWidth={isActive ? 2.1 : 1.6}
                className={isActive ? 'text-content-strong' : 'text-content-subtle'}
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
