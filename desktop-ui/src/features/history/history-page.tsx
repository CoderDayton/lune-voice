import { PageIntroPanel } from "../../components/ui/page-intro-panel";
import { HistoryRail } from "./components/history-rail";
import { HistorySessionList } from "./components/history-session-list";
import { HistorySummary } from "./components/history-summary";

export function HistoryPage() {
  return (
    <div className="page-stack">
      <PageIntroPanel
        kicker="Your sessions"
        title="Find past recordings and the text they produced."
        description="Search what you said, reopen recent dictation, and spot phrases that should become permanent word fixes."
      />

      <HistorySummary />

      <section className="layout-grid xl:grid-cols-[1.18fr_0.82fr]">
        <HistorySessionList />
        <HistoryRail />
      </section>
    </div>
  );
}
