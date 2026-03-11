import { useStore } from "@tanstack/react-store";
import { Link } from "@tanstack/react-router";
import { buttonClassName } from "../../../components/ui/button";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
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

  return (
    <div className="page-grid-hero">
      <Panel tone="hero" className="panel-pad-lg relative overflow-hidden">
        <p className="app-kicker-accent">Voice typing, simplified</p>
        <h2 className="hero-title mt-3">
          Speak naturally. Keep the final text polished.
        </h2>
        <p className="body-copy-strong mt-4 max-w-[var(--size-measure-wide)]">
          Lune Voice helps you record quickly, fix common wording automatically, and send clean text wherever you are working.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/history" className={buttonClassName("primary")}>
            View recent sessions
          </Link>
          <Link to="/recording" className={buttonClassName("secondary")}>
            Set up recording
          </Link>
        </div>
      </Panel>

      <Panel className="panel-pad-lg">
        <div className="header-row">
          <div>
            <Pill>Session snapshot</Pill>
            <h3 className="section-title mt-4">Current operating picture</h3>
            <p className="body-copy-soft mt-2">
              A quick read on what is ready right now before you start dictating.
            </p>
          </div>
          <Pill>{runtime.serviceStatus}</Pill>
        </div>

        <div className="hero-summary-grid mt-6">
          <div className="stat-tile">
            <div className="stat-label">Capture</div>
            <div className="stat-value mt-3">{runtime.loaded ? "Connected" : "Loading"}</div>
            <p className="stat-note mt-1">Tauri IPC status for the desktop backend.</p>
          </div>
          <div className="stat-tile">
            <div className="stat-label">Speech model</div>
            <div className="stat-value mt-3">{speechModel}</div>
            <p className="stat-note mt-1">Matches the transcription mode selected on your recording setup page.</p>
          </div>
          <div className="stat-tile">
            <div className="stat-label">Text delivery</div>
            <div className="stat-value mt-3">{deliveryMode}</div>
            <p className="stat-note mt-1">Reflects how finished text will be inserted into the app you are using.</p>
          </div>
          <div className="stat-tile">
            <div className="stat-label">Response time</div>
            <div className="stat-value mt-3">{averageResponseTime}</div>
            <p className="stat-note mt-1">Average recorded session duration in the local archive.</p>
          </div>
        </div>
      </Panel>
    </div>
  );
}
