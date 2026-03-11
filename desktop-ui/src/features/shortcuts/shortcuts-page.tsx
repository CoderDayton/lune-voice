import { useStore } from "@tanstack/react-store";
import { SelectField } from "../../components/ui/field";
import { PageIntroPanel } from "../../components/ui/page-intro-panel";
import { Panel } from "../../components/ui/panel";
import { StatTile } from "../../components/ui/stat-tile";
import {
  saveWorkspaceSetting,
  summarizeDictationMode,
  summarizeStylePreset,
  workspaceStore,
} from "../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../setup/components/settings-section";
import {
  deliveryMethods,
  labelForActivationMode,
  stylePresets,
} from "../setup/data/settings-data";

export function ShortcutsPage() {
  const settings = useStore(workspaceStore, (state) => state.settings);

  return (
    <div className="page-stack">
      <PageIntroPanel
        kicker="Shortcuts"
        title="Control recording quickly and send text where you need it."
        description="Keep the keyboard side simple. This page focuses on how you start voice typing, stop it, and place the final text into the app you are using."
      />

      <section className="layout-grid md:grid-cols-3">
        <StatTile
          label="Start and stop"
          value={settings.hotkeyToggle}
          note="The main shortcut for entering and leaving voice typing."
        />
        <StatTile
          label="Cancel"
          value={settings.hotkeyCancel}
          note="Quickly discard the current recording without sending any text."
        />
        <StatTile
          label="Send text"
          value={summarizeDictationMode(settings.dictationMode)}
          note="Choose whether Lune Voice pastes finished text or types it out like keystrokes."
        />
      </section>

      <section className="layout-grid">
        <SettingsSection title="Keyboard controls" description="Use fast commands without opening menus in the middle of work.">
          <SettingsRow label="Start or stop recording" description="The main shortcut used for voice typing." control={<span className="app-pill font-mono">{settings.hotkeyToggle}</span>} />
          <SettingsRow label="Cancel current recording" description="Discard the current draft and return to idle." control={<span className="app-pill font-mono">{settings.hotkeyCancel}</span>} />
          <SettingsRow label="Recording style" description="This changes whether the shortcut toggles recording or only works while held." control={<span className="app-pill">{labelForActivationMode(settings.activationMode).split(" ")[0]}</span>} />
        </SettingsSection>

        <SettingsSection title="Sending text" description="Choose how finished text is inserted into the app you are currently using.">
          <SettingsRow
            label="Send method"
            description="Paste is simplest. Typing keystrokes can work better in some apps."
            control={
              <SelectField
                value={settings.dictationMode}
                onChange={(event) => void saveWorkspaceSetting("dictationMode", event.target.value as typeof settings.dictationMode)}
                className="min-w-56"
              >
                {deliveryMethods.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </SelectField>
            }
          />
          <SettingsRow
            label="Writing style"
            description="Pick the tone used when final cleanup is applied."
            control={
              <SelectField
                value={settings.stylePreset}
                onChange={(event) => void saveWorkspaceSetting("stylePreset", event.target.value)}
                className="min-w-52"
              >
                {stylePresets.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </SelectField>
            }
          />
        </SettingsSection>

        <Panel className="panel-pad-lg xl:col-span-2">
          <p className="app-kicker-accent">Everyday use</p>
          <div className="layout-grid mt-4 md:grid-cols-3">
            <div className="stat-tile">
              <h3 className="card-title-sm">For long notes</h3>
              <p className="stat-note mt-2">Use toggle recording and paste into your current app.</p>
            </div>
            <div className="stat-tile">
              <h3 className="card-title-sm">For quick commands</h3>
              <p className="stat-note mt-2">Use hold-to-talk so recording starts only while the shortcut is pressed.</p>
            </div>
            <div className="stat-tile">
              <h3 className="card-title-sm">For code editors</h3>
              <p className="stat-note mt-2">Current writing style: {summarizeStylePreset(settings.stylePreset)}.</p>
            </div>
          </div>
        </Panel>
      </section>
    </div>
  );
}
