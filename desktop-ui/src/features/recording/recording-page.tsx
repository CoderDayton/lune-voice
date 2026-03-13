import { useStore } from "@tanstack/react-store";
import { SelectField, TextField } from "../../components/ui/field";
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
import {
  activationModes,
  asrBackends,
  labelForActivationMode,
  llmModels,
} from "../setup/data/settings-data";

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
                className="min-w-52"
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
              />
            }
          />
          <SettingsRow label="Pause sensitivity" description="Lower numbers wait longer before deciding you are done speaking." control={<span className="badge badge-outline badge-sm font-mono">{settings.speechThreshold}</span>} />
        </SettingsSection>

        <SettingsSection title="Transcription quality" description="Control how your speech is turned into text.">
          <SettingsRow
            label="Voice mode"
            description="Balanced works well for most people. Compact uses less memory."
            control={
              <SelectField
                value={settings.asrBackend}
                onChange={(event) => void saveWorkspaceSetting("asrBackend", event.target.value as typeof settings.asrBackend)}
                className="min-w-64"
              >
                {asrBackends.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </SelectField>
            }
          />
          {settings.asrBackend === "OpenAiCompatible" && (
            <>
              <SettingsRow
                label="API endpoint"
                description="Base URL for the OpenAI-compatible transcription API."
                control={
                  <TextField
                    value={settings.openaiAsrBaseUrl}
                    onChange={(event) => void saveWorkspaceSetting("openaiAsrBaseUrl", event.target.value)}
                    className="min-w-64"
                    placeholder="https://api.openai.com/v1"
                  />
                }
              />
              <SettingsRow
                label="API key"
                description="Your API key. Stored locally and never shared."
                control={
                  <TextField
                    type="password"
                    value={settings.openaiAsrApiKey}
                    onChange={(event) => void saveWorkspaceSetting("openaiAsrApiKey", event.target.value)}
                    className="min-w-64"
                    placeholder="sk-..."
                  />
                }
              />
              <SettingsRow
                label="Model"
                description="Transcription model name, e.g. whisper-1."
                control={
                  <TextField
                    value={settings.openaiAsrModel}
                    onChange={(event) => void saveWorkspaceSetting("openaiAsrModel", event.target.value)}
                    className="min-w-48"
                    placeholder="whisper-1"
                  />
                }
              />
            </>
          )}
          <SettingsRow
            label="Use hardware acceleration"
            description="When available, this can make transcription finish faster."
            control={<Toggle checked={settings.useGpu} onChange={() => void saveWorkspaceSetting("useGpu", !settings.useGpu)} />}
          />
          <SettingsRow
            label="Word-level alignment"
            description="Use forced aligner for precise word timestamps and chunk merging."
            control={
              <Toggle
                checked={settings.alignerEnabled}
                onChange={() => void saveWorkspaceSetting("alignerEnabled", !settings.alignerEnabled)}
              />
            }
          />
        </SettingsSection>

        <SettingsSection title="Smart cleanup" description="Give uncertain words a second look before text is sent to your app.">
          <SettingsRow
            label="Automatic cleanup"
            description="Use a local language model to tidy up uncertain phrases."
            control={
              <Toggle
                checked={settings.llmCorrection}
                onChange={() => void saveWorkspaceSetting("llmCorrection", !settings.llmCorrection)}
              />
            }
          />
          <SettingsRow
            label="Cleanup model"
            control={
              <SelectField
                value={settings.llmBackend}
                onChange={(event) => void saveWorkspaceSetting("llmBackend", event.target.value as typeof settings.llmBackend)}
                className="min-w-64"
              >
                {llmModels.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </SelectField>
            }
          />
          {settings.llmBackend === "OpenAiCompatible" && (
            <>
              <SettingsRow
                label="API endpoint"
                description="Base URL for the OpenAI-compatible chat completion API."
                control={
                  <TextField
                    value={settings.openaiLlmBaseUrl}
                    onChange={(event) => void saveWorkspaceSetting("openaiLlmBaseUrl", event.target.value)}
                    className="min-w-64"
                    placeholder="https://api.openai.com/v1"
                  />
                }
              />
              <SettingsRow
                label="API key"
                description="Your API key. Stored locally and never shared."
                control={
                  <TextField
                    type="password"
                    value={settings.openaiLlmApiKey}
                    onChange={(event) => void saveWorkspaceSetting("openaiLlmApiKey", event.target.value)}
                    className="min-w-64"
                    placeholder="sk-..."
                  />
                }
              />
              <SettingsRow
                label="Model"
                description="Chat completion model name, e.g. gpt-4o-mini."
                control={
                  <TextField
                    value={settings.openaiLlmModel}
                    onChange={(event) => void saveWorkspaceSetting("openaiLlmModel", event.target.value)}
                    className="min-w-48"
                    placeholder="gpt-4o-mini"
                  />
                }
              />
            </>
          )}
          <SettingsRow
            label="Cleanup sensitivity"
            description="Words below this score are candidates for cleanup."
            control={<span className="badge badge-outline badge-sm font-mono">{settings.confidenceThreshold}</span>}
          />
        </SettingsSection>

        <Panel className="p-6">
          <p className="app-kicker-accent">Good default</p>
          <h3 className="card-title-text mt-3">Recommended starting setup</h3>
          <div className="flex flex-col gap-4 mt-4">
            <p>Use <span className="font-semibold">Toggle</span> if you dictate longer notes or emails.</p>
            <p>Keep <span className="font-semibold">Balanced</span> selected unless your computer is memory-constrained.</p>
            <p>Leave <span className="font-semibold">Smart cleanup</span> on so names and uncertain phrases get a second pass.</p>
          </div>
        </Panel>
      </section>
    </div>
  );
}
