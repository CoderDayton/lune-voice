import { Trans } from "@lingui/react/macro";
import { useStore } from "@tanstack/react-store";
import { Link } from "@tanstack/react-router";
import { buttonClassName } from "../../../components/ui/button";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { StatTile } from "../../../components/ui/stat-tile";
import {
  summarizeAsrBackend,
  summarizeDictationMode,
  workspaceStore,
} from "../../../store/workspace-store";

export function OverviewHero() {
  const settings = useStore(workspaceStore, (state) => state.settings);
  const runtime = useStore(workspaceStore, (state) => state.runtime);
  const historySessions = useStore(workspaceStore, (state) => state.historySessions);
  const speechModel = summarizeAsrBackend(settings.asrBackend);
  const deliveryMode = summarizeDictationMode(settings.dictationMode);
  const averageResponseTime = historySessions.length > 0
    ? `${(historySessions.reduce((sum, item) => sum + item.seconds, 0) / historySessions.length).toFixed(2)}s`
    : "Waiting";
  const summaryTiles = [
    {
      label: "Capture",
      value: runtime.loaded ? "Connected" : "Loading",
      note: "Tauri IPC status for the desktop backend.",
    },
    {
      label: "Speech model",
      value: speechModel,
      note: "Matches the transcription mode selected on your recording setup page.",
    },
    {
      label: "Text delivery",
      value: deliveryMode,
      note: "Reflects how finished text will be inserted into the app you are using.",
    },
    {
      label: "Response time",
      value: averageResponseTime,
      note: "Average recorded session duration in the local archive.",
    },
  ];

  return (
    <div className="layout-grid xl:grid-cols-[1.35fr_0.9fr]">
      <Panel tone="hero" className="p-4 sm:p-6 relative overflow-hidden">
        <p className="app-kicker-accent"><Trans>Voice typing, simplified</Trans></p>
        <h2 className="hero-title mt-3">
          <Trans>Speak naturally. Keep the final text polished.</Trans>
        </h2>
        <p className="body-copy-strong mt-4 max-w-[var(--size-measure-wide)]">
          <Trans>Lune Voice helps you record quickly, fix common wording automatically, and send clean text wherever you are working.</Trans>
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/history" className={buttonClassName("primary")}>
            <Trans>View recent sessions</Trans>
          </Link>
          <Link to="/settings" className={buttonClassName("secondary")}>
            <Trans>Open settings</Trans>
          </Link>
        </div>
      </Panel>

      <Panel className="p-4 sm:p-6">
        <div className="header-row">
          <div>
            <Pill><Trans>Session snapshot</Trans></Pill>
            <h3 className="section-title mt-4"><Trans>Current operating picture</Trans></h3>
            <p className="body-copy-soft mt-2">
              <Trans>A quick read on what is ready right now before you start dictating.</Trans>
            </p>
          </div>
          <Pill>{runtime.serviceStatus}</Pill>
        </div>

        <div className="layout-grid-tight mt-6 md:grid-cols-2">
          {summaryTiles.map((tile) => (
            <StatTile
              key={tile.label}
              label={tile.label}
              value={tile.value}
              note={tile.note}
              noteClassName="mt-1"
            />
          ))}
        </div>
      </Panel>
    </div>
  );
}
