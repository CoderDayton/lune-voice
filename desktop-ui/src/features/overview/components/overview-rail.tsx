import { useMemo } from "react";
import { Trans } from "@lingui/react/macro";
import { useStore } from "@tanstack/react-store";
import { Link } from "@tanstack/react-router";
import { buttonClassName } from "../../../components/ui/button";
import { Panel } from "../../../components/ui/panel";
import {
  summarizeAsrBackend,
  summarizeLlmBackend,
  workspaceStore,
} from "../../../store/workspace-store";

export function OverviewRail() {
  const settings = useStore(workspaceStore, (state) => state.settings);
  const runtime = useStore(workspaceStore, (state) => state.runtime);
  const signals = useMemo(
    () => [
      {
        title: "Desktop backend",
        value: runtime.serviceStatus,
        note: "Reads and writes now flow through the local Tauri backend.",
      },
      {
        title: "ASR posture",
        value: summarizeAsrBackend(settings.asrBackend),
        note: "This reflects the speech backend currently saved in your desktop settings.",
      },
      {
        title: "Correction chain",
        value: settings.llmCorrection ? summarizeLlmBackend(settings.llmBackend) : "Cleanup off",
        note: "Low-confidence words route through cleanup only when the confidence gate is enabled.",
      },
    ],
    [runtime.serviceStatus, settings.asrBackend, settings.llmBackend, settings.llmCorrection],
  );

  return (
    <div className="stack-section">
      <Panel className="p-4 sm:p-6">
        <p className="app-kicker-accent"><Trans>Workspace posture</Trans></p>
        <div className="rail-list mt-4">
          {signals.map((item) => (
            <div key={item.title} className="rail-row">
              <div className="stack-copy-tight">
                <div className="supporting-label">{item.title}</div>
                <div className="body-copy-soft mt-1">{item.note}</div>
              </div>
              <div className="value-display-sm shrink-0 text-right">{item.value}</div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel className="p-4 sm:p-6">
        <p className="app-kicker-accent"><Trans>Next move</Trans></p>
        <h3 className="section-title mt-3"><Trans>Normalize repeated corrections.</Trans></h3>
        <p className="body-copy-soft mt-2">
          <Trans>The fastest visible quality gain now is tightening the dictionary layer around common house terms and code nouns.</Trans>
        </p>
        <div className="mt-5">
          <Link to="/dictionary" className={buttonClassName("primary")}>
            <Trans>Open dictionary</Trans>
          </Link>
        </div>
      </Panel>
    </div>
  );
}
