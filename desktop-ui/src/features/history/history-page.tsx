import { Panel } from "../../components/ui/panel";
import { HistoryRail } from "./components/history-rail";
import { HistorySessionList } from "./components/history-session-list";
import { HistorySummary } from "./components/history-summary";

export function HistoryPage() {
  return (
    <div className="page-stack">
      <Panel className="panel-pad-lg">
        <div className="page-intro">
          <p className="app-kicker-accent">Your sessions</p>
          <h2 className="page-title mt-3">Find past recordings and the text they produced.</h2>
          <p className="body-copy mt-3">
            Search what you said, reopen recent dictation, and spot phrases that should become permanent word fixes.
          </p>
        </div>
      </Panel>

      <HistorySummary />

      <section className="page-grid-history">
        <HistorySessionList />
        <HistoryRail />
      </section>
    </div>
  );
}
