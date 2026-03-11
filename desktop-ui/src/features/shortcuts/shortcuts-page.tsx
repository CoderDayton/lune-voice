import { useStore } from "@tanstack/react-store";
import { SelectField } from "../../components/ui/field";
import { Panel } from "../../components/ui/panel";
import { workspaceActions, workspaceStore } from "../../store/workspace-store";
import { SettingsRow, SettingsSection } from "../setup/components/settings-section";
import { deliveryMethods, stylePresets } from "../setup/data/settings-data";

export function ShortcutsPage() {
  const settings = useStore(workspaceStore, (state) => state.settings);

  return (
    <div className="page-stack">
      <Panel className="panel-pad-lg">
        <div className="page-intro">
          <p className="app-kicker-accent">Shortcuts</p>
          <h2 className="page-title mt-3">Control recording quickly and send text where you need it.</h2>
          <p className="body-copy mt-3">
            Keep the keyboard side simple. This page focuses on how you start voice typing, stop it, and place the final text into the app you are using.
          </p>
        </div>
      </Panel>

      <section className="page-grid-three">
        <Panel className="panel-pad-md">
          <div className="stat-label">Start and stop</div>
          <div className="stat-value mt-3">Ctrl+Space</div>
          <p className="stat-note mt-2">The main shortcut for entering and leaving voice typing.</p>
        </Panel>
        <Panel className="panel-pad-md">
          <div className="stat-label">Cancel</div>
          <div className="stat-value mt-3">Escape</div>
          <p className="stat-note mt-2">Quickly discard the current recording without sending any text.</p>
        </Panel>
        <Panel className="panel-pad-md">
          <div className="stat-label">Send text</div>
          <div className="stat-value mt-3">{settings.deliveryMethod.startsWith("Paste") ? "Paste" : "Type"}</div>
          <p className="stat-note mt-2">Choose whether Lune Voice pastes finished text or types it out like keystrokes.</p>
        </Panel>
      </section>

      <section className="page-grid-two">
        <SettingsSection title="Keyboard controls" description="Use fast commands without opening menus in the middle of work.">
          <SettingsRow label="Start or stop recording" description="The main shortcut used for voice typing." control={<span className="app-pill font-mono">Ctrl+Space</span>} />
          <SettingsRow label="Cancel current recording" description="Discard the current draft and return to idle." control={<span className="app-pill font-mono">Escape</span>} />
          <SettingsRow label="Recording style" description="This changes whether the shortcut toggles recording or only works while held." control={<span className="app-pill">{settings.activationMode}</span>} />
        </SettingsSection>

        <SettingsSection title="Sending text" description="Choose how finished text is inserted into the app you are currently using.">
          <SettingsRow
            label="Send method"
            description="Paste is simplest. Typing keystrokes can work better in some apps."
            control={
              <SelectField
                value={settings.deliveryMethod}
                onChange={(event) => workspaceActions.updateSetting("deliveryMethod", event.target.value)}
                className="min-w-56"
              >
                {deliveryMethods.map((option) => (
                  <option key={option}>{option}</option>
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
                onChange={(event) => workspaceActions.updateSetting("stylePreset", event.target.value)}
                className="min-w-52"
              >
                {stylePresets.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </SelectField>
            }
          />
        </SettingsSection>

        <Panel className="panel-pad-lg xl:col-span-2">
          <p className="app-kicker-accent">Everyday use</p>
          <div className="tile-grid-three-wide mt-4">
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
              <p className="stat-note mt-2">Typing keystrokes can be more reliable when paste handling is restricted.</p>
            </div>
          </div>
        </Panel>
      </section>
    </div>
  );
}
