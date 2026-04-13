import { useStore } from "@tanstack/react-store";
import { workspaceStore } from "../../store/workspace-store";
import { AppearanceSection } from "./sections/appearance-section";
import { InputSection } from "./sections/input-section";
import { TranscriptionSection } from "./sections/transcription-section";
import { ShortcutsSection } from "./sections/shortcuts-section";
import { OutputSection } from "./sections/output-section";
import { ModelsSection } from "./sections/models-section";

export function SettingsPage() {
  const settings = useStore(workspaceStore, (s) => s.settings);

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
