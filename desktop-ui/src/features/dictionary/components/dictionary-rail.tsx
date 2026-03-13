import { useStore } from "@tanstack/react-store";
import { Panel } from "../../../components/ui/panel";
import { workspaceStore } from "../../../store/workspace-store";

export function DictionaryRail() {
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);

  return (
    <div className="stack-section">
      <Panel className="p-5">
        <p className="app-kicker-accent">Rulebook</p>
        <h3 className="card-title-text mt-3">Keep repeated corrections out of the review loop.</h3>
        <p className="body-copy-soft mt-2">
          Every saved term helps Lune Voice fix names and phrases automatically before text reaches your app.
        </p>
      </Panel>

      <Panel className="p-5">
        <p className="app-kicker-accent">Current posture</p>
        <div className="flex flex-col gap-4 mt-4">
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="opacity-50">Active rules</span>
            <span className="value-display-sm">{entries.length}</span>
          </div>
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="opacity-50">Default priority</span>
            <span className="font-medium opacity-80">5</span>
          </div>
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="opacity-50">Applies</span>
            <span className="font-medium opacity-80">Before sending text</span>
          </div>
        </div>
      </Panel>
    </div>
  );
}
