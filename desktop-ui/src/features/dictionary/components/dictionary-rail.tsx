import { useStore } from "@tanstack/react-store";
import { Panel } from "../../../components/ui/panel";
import { workspaceStore } from "../../../store/workspace-store";
import { Trans } from "@lingui/react/macro";

export function DictionaryRail() {
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);

  return (
    <div className="stack-section">
      <Panel className="p-4 sm:p-5">
        <p className="app-kicker-accent"><Trans>Rulebook</Trans></p>
        <h3 className="card-title-text mt-3"><Trans>Keep repeated corrections out of the review loop.</Trans></h3>
        <p className="body-copy-soft mt-2">
          <Trans>Every saved term helps Lune Voice fix names and phrases automatically before text reaches your app.</Trans>
        </p>
      </Panel>

      <Panel className="p-4 sm:p-5">
        <p className="app-kicker-accent"><Trans>Current posture</Trans></p>
        <div className="flex flex-col gap-4 mt-4">
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="opacity-50"><Trans>Active rules</Trans></span>
            <span className="value-display-sm">{entries.length}</span>
          </div>
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="opacity-50"><Trans>Default priority</Trans></span>
            <span className="font-medium opacity-80">5</span>
          </div>
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="opacity-50"><Trans>Applies</Trans></span>
            <span className="font-medium opacity-80"><Trans>Before sending text</Trans></span>
          </div>
        </div>
      </Panel>
    </div>
  );
}
