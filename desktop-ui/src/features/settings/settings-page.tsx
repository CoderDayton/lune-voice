import { useStore } from "@tanstack/react-store";
import { PageIntroPanel } from "../../components/ui/page-intro-panel";
import { StatTile } from "../../components/ui/stat-tile";
import {
  summarizeAsrBackend,
  summarizeDictationMode,
  workspaceStore,
} from "../../store/workspace-store";
import { labelForActivationMode } from "../setup/data/settings-data";
import { InputSection } from "./sections/input-section";
import { TranscriptionSection } from "./sections/transcription-section";
import { CleanupSection } from "./sections/cleanup-section";
import { ShortcutsSection } from "./sections/shortcuts-section";
import { OutputSection } from "./sections/output-section";

export function SettingsPage() {
  const settings = useStore(workspaceStore, (state) => state.settings);

  return (
    <div className="page-stack">
      <PageIntroPanel
        kicker="Settings"
        title="Voice input, recognition, output, and shortcuts in one place."
        description="Configure how Lune Voice listens, transcribes, cleans up text, and delivers it to the app you are using."
      />

      <section className="layout-grid sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label="Recording style"
          value={labelForActivationMode(settings.activationMode).split(" ")[0]}
          note="Toggle or hold-to-talk."
        />
        <StatTile
          label="Voice mode"
          value={summarizeAsrBackend(settings.asrBackend)}
          note="Active speech recognition backend."
        />
        <StatTile
          label="Smart cleanup"
          value={settings.llmCorrection ? "On" : "Off"}
          note="Automatic correction of uncertain words."
        />
        <StatTile
          label="Text delivery"
          value={summarizeDictationMode(settings.dictationMode)}
          note="How finished text reaches your app."
        />
      </section>

      <section className="layout-grid">
        <InputSection settings={settings} />
        <TranscriptionSection settings={settings} />
        <CleanupSection settings={settings} />
        <ShortcutsSection settings={settings} />
        <OutputSection settings={settings} />
      </section>
    </div>
  );
}
