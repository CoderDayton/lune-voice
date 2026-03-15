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
import { useLingui } from "@lingui/react/macro";

export function SettingsPage() {
  const { t } = useLingui();
  const settings = useStore(workspaceStore, (state) => state.settings);

  return (
    <div className="page-stack">
      <PageIntroPanel
        kicker="Settings"
        title={t`Voice input, recognition, output, and shortcuts in one place.`}
        description={t`Configure how Lune Voice listens, transcribes, cleans up text, and delivers it to the app you are using.`}
      />

      <section className="layout-grid sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label={t`Recording style`}
          value={labelForActivationMode(settings.activationMode).split(" ")[0]}
          note={t`Toggle or hold-to-talk.`}
        />
        <StatTile
          label={t`Voice mode`}
          value={summarizeAsrBackend(settings.asrBackend)}
          note={t`Active speech recognition backend.`}
        />
        <StatTile
          label={t`Smart cleanup`}
          value={settings.llmCorrection ? "On" : "Off"}
          note={t`Automatic correction of uncertain words.`}
        />
        <StatTile
          label={t`Text delivery`}
          value={summarizeDictationMode(settings.dictationMode)}
          note={t`How finished text reaches your app.`}
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
