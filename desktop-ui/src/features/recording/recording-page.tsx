import { useStore } from "@tanstack/react-store";
import { SelectField } from "../../components/ui/field";
import { Panel } from "../../components/ui/panel";
import { Toggle } from "../../components/ui/toggle";
import { workspaceActions, workspaceStore } from "../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../setup/components/settings-section";
import { asrBackends, llmModels } from "../setup/data/settings-data";

export function RecordingPage() {
  const settings = useStore(workspaceStore, (state) => state.settings);

  return (
    <div className="page-stack">
      <Panel className="panel-pad-lg">
        <div className="page-intro">
          <p className="app-kicker-accent">Recording</p>
          <h2 className="page-title mt-3">Set how Lune Voice listens and cleans up what you say.</h2>
          <p className="body-copy mt-3">
            This is the main setup page for voice typing. Choose the listening style, transcription quality, and cleanup level that feel best for your everyday workflow.
          </p>
        </div>
      </Panel>

      <section className="page-grid-three">
        <Panel className="panel-pad-md">
          <div className="stat-label">Recording style</div>
          <div className="stat-value mt-3">{settings.activationMode}</div>
          <p className="stat-note mt-2">Choose whether recording stays on until you stop it or only while a key is held.</p>
        </Panel>
        <Panel className="panel-pad-md">
          <div className="stat-label">Voice mode</div>
          <div className="stat-value mt-3">{settings.asrBackend.split(" ")[0]}</div>
          <p className="stat-note mt-2">Pick the listening mode that best matches your device and speed preference.</p>
        </Panel>
        <Panel className="panel-pad-md">
          <div className="stat-label">Smart cleanup</div>
          <div className="stat-value mt-3">{settings.llmCorrection ? "On" : "Off"}</div>
          <p className="stat-note mt-2">Turn on automatic cleanup for words that sound uncertain or need a second pass.</p>
        </Panel>
      </section>

      <section className="page-grid-two">
        <SettingsSection title="How recording starts" description="Choose how quickly you want to enter and leave voice typing.">
          <SettingsRow
            label="Recording mode"
            description="Toggle is easiest for longer dictation. Hold is better for quick bursts."
            control={
              <SelectField
                value={settings.activationMode}
                onChange={(event) => workspaceActions.updateSetting("activationMode", event.target.value)}
                className="min-w-52"
              >
                <option value="Toggle">Toggle to start and stop</option>
                <option value="Hold">Hold while speaking</option>
              </SelectField>
            }
          />
          <SettingsRow
            label="Adaptive pause detection"
            description="Lune Voice learns your pause rhythm so it does not cut you off too early."
            control={
              <Toggle
                checked={settings.adaptiveVad}
                onChange={() => workspaceActions.updateSetting("adaptiveVad", !settings.adaptiveVad)}
              />
            }
          />
          <SettingsRow label="Pause sensitivity" description="Lower numbers wait longer before deciding you are done speaking." control={<span className="app-pill font-mono">{settings.speechThreshold}</span>} />
        </SettingsSection>

        <SettingsSection title="Transcription quality" description="Control how your speech is turned into text.">
          <SettingsRow
            label="Voice mode"
            description="Balanced works well for most people. Compact uses less memory."
            control={
              <SelectField
                value={settings.asrBackend}
                onChange={(event) => workspaceActions.updateSetting("asrBackend", event.target.value)}
                className="min-w-64"
              >
                {asrBackends.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </SelectField>
            }
          />
          <SettingsRow
            label="Use hardware acceleration"
            description="When available, this can make transcription finish faster."
            control={<Toggle checked={settings.useGpu} onChange={() => workspaceActions.updateSetting("useGpu", !settings.useGpu)} />}
          />
        </SettingsSection>

        <SettingsSection title="Smart cleanup" description="Give uncertain words a second look before text is sent to your app.">
          <SettingsRow
            label="Automatic cleanup"
            description="Use a local language model to tidy up uncertain phrases."
            control={
              <Toggle
                checked={settings.llmCorrection}
                onChange={() => workspaceActions.updateSetting("llmCorrection", !settings.llmCorrection)}
              />
            }
          />
          <SettingsRow
            label="Cleanup model"
            control={
              <SelectField
                value={settings.llmModel}
                onChange={(event) => workspaceActions.updateSetting("llmModel", event.target.value)}
                className="min-w-64"
              >
                {llmModels.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </SelectField>
            }
          />
          <SettingsRow
            label="Cleanup sensitivity"
            description="Words below this score are candidates for cleanup."
            control={<span className="app-pill font-mono">{settings.confidenceThreshold}</span>}
          />
        </SettingsSection>

        <Panel className="panel-pad-lg">
          <p className="app-kicker-accent">Good default</p>
          <h3 className="card-title mt-3">Recommended starting setup</h3>
          <div className="insight-stack mt-4">
            <p>Use <span className="font-semibold text-copy-50">Toggle</span> if you dictate longer notes or emails.</p>
            <p>Keep <span className="font-semibold text-copy-50">Balanced</span> selected unless your computer is memory-constrained.</p>
            <p>Leave <span className="font-semibold text-copy-50">Smart cleanup</span> on so names and uncertain phrases get a second pass.</p>
          </div>
        </Panel>
      </section>
    </div>
  );
}
