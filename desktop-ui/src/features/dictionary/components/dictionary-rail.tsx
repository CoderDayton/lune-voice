import { useStore } from "@tanstack/react-store";
import { Panel } from "../../../components/ui/panel";
import { workspaceStore } from "../../../store/workspace-store";

export function DictionaryRail() {
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);

  return (
    <div className="stack-section">
      <Panel className="panel-pad-md">
        <p className="app-kicker-accent">Rulebook</p>
        <h3 className="card-title mt-3">Keep repeated corrections out of the review loop.</h3>
        <p className="body-copy-soft mt-2">
          Every saved term helps Lune Voice fix names and phrases automatically before text reaches your app.
        </p>
      </Panel>

      <Panel className="panel-pad-md">
        <p className="app-kicker-accent">Current posture</p>
        <div className="insight-stack mt-4">
          <div className="sidebar-list-row">
            <span className="sidebar-key">Active rules</span>
            <span className="value-display-sm text-copy-50">{entries.length}</span>
          </div>
          <div className="sidebar-list-row">
            <span className="sidebar-key">Default priority</span>
            <span className="sidebar-value">5</span>
          </div>
          <div className="sidebar-list-row">
            <span className="sidebar-key">Applies</span>
            <span className="sidebar-value">Before sending text</span>
          </div>
        </div>
      </Panel>
    </div>
  );
}
