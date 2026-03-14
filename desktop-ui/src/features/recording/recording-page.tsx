import { useStore } from "@tanstack/react-store";
import { SelectField } from "../../components/ui/field";
import { PageIntroPanel } from "../../components/ui/page-intro-panel";
import { Panel } from "../../components/ui/panel";
import { StatTile } from "../../components/ui/stat-tile";
import { Toggle } from "../../components/ui/toggle";
import {
  saveWorkspaceSetting,
  summarizeAsrBackend,
  workspaceStore,
} from "../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../setup/components/settings-section";
import { activationModes, labelForActivationMode } from "../setup/data/settings-data";
import { CleanupSection, TranscriptionSection } from "./recording-sections";

export function RecordingPage() {
  const settings = useStore(workspaceStore, (state) => state.settings);

  return (
    <div className="page-stack">
      <PageIntroPanel
        kicker="Recording"
        title="Set how Lune Voice listens and cleans up what you say."
        description="This is the main setup page for voice typing. Choose the listening style, transcription quality, and cleanup level that feel best for your everyday workflow."
      />

      <section className="layout-grid md:grid-cols-3">
        <StatTile
          label="Recording style"
          value={labelForActivationMode(settings.activationMode).split(" ")[0]}
          note="Choose whether recording stays on until you stop it or only while a key is held."
        />
        <StatTile
          label="Voice mode"
          value={summarizeAsrBackend(settings.asrBackend)}
          note="Pick the listening mode that best matches your device and speed preference."
        />
        <StatTile
          label="Smart cleanup"
          value={settings.llmCorrection ? "On" : "Off"}
          note="Turn on automatic cleanup for words that sound uncertain or need a second pass."
        />
      </section>

      <section className="layout-grid">
        <SettingsSection title="How recording starts" description="Choose how quickly you want to enter and leave voice typing.">
          <SettingsRow
            label="Recording mode"
            description="Toggle is easiest for longer dictation. Hold is better for quick bursts."
            control={
              <SelectField
                value={settings.activationMode}
                onChange={(event) => void saveWorkspaceSetting("activationMode", event.target.value as typeof settings.activationMode)}
                className="w-full sm:w-auto sm:min-w-52"
              >
                {activationModes.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </SelectField>
            }
          />
          <SettingsRow
            label="Adaptive pause detection"
            description="Lune Voice learns your pause rhythm so it does not cut you off too early."
            control={
              <Toggle
                checked={settings.adaptiveVad}
                onChange={() => void saveWorkspaceSetting("adaptiveVad", !settings.adaptiveVad)}
                aria-label="Adaptive pause detection"
              />
            }
          />
          <SettingsRow label="Pause sensitivity" description="Lower numbers wait longer before deciding you are done speaking." control={<span className="badge badge-outline badge-sm font-mono">{settings.speechThreshold}</span>} />
        </SettingsSection>

        <TranscriptionSection settings={settings} />
        <CleanupSection settings={settings} />

        <Panel className="p-4 md:p-6">
          <p className="app-kicker-accent">Good default</p>
          <h3 className="card-title-text mt-3">Recommended starting setup</h3>
          <div className="flex flex-col gap-3 sm:gap-4 mt-4">
            <p>Use <span className="font-semibold">Toggle</span> if you dictate longer notes or emails.</p>
            <p>Keep <span className="font-semibold">Balanced</span> selected unless your computer is memory-constrained.</p>
            <p>Leave <span className="font-semibold">Smart cleanup</span> on so names and uncertain phrases get a second pass.</p>
          </div>
        </Panel>
      </section>
    </div>
  );
}
