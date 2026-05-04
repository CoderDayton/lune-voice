import { useStore } from '@tanstack/react-store';
import { workspaceStore } from '../../store/workspace-store';
import { AppearanceSection } from './sections/appearance-section';
import { InputSection } from './sections/input-section';
import { ModelsSection } from './sections/models-section';
import { OutputSection } from './sections/output-section';
import { ShortcutsSection } from './sections/shortcuts-section';
import { TranscriptionSection } from './sections/transcription-section';

export function SettingsPage() {
  const settings = useStore(workspaceStore, (s) => s.settings);
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

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

  return (
    <div className="page-stack">
      <AppearanceSection />
      <InputSection settings={settings} />
      <TranscriptionSection settings={settings} />
      <OutputSection settings={settings} />
      <ModelsSection />
      <ShortcutsSection settings={settings} />
    </div>
  );
}
