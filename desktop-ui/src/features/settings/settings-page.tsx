import { useStore } from "@tanstack/react-store";
import { workspaceStore } from "../../store/workspace-store";
import { InputSection } from "./sections/input-section";
import { TranscriptionSection } from "./sections/transcription-section";
import { CleanupSection } from "./sections/cleanup-section";
import { ShortcutsSection } from "./sections/shortcuts-section";
import { OutputSection } from "./sections/output-section";

export function SettingsPage() {
  const settings = useStore(workspaceStore, (s) => s.settings);

  return (
    <div className="page-stack">
      <InputSection settings={settings} />
      <TranscriptionSection settings={settings} />
      <CleanupSection settings={settings} />
      <OutputSection settings={settings} />
      <ShortcutsSection settings={settings} />
    </div>
  );
}
