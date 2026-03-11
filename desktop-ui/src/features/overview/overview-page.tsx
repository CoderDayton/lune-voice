import { ActivityFeed } from "./components/activity-feed";
import { OverviewHero } from "./components/overview-hero";
import { OverviewRail } from "./components/overview-rail";
import { OverviewStats } from "./components/overview-stats";

export function OverviewPage() {
  return (
    <div className="page-stack">
      <OverviewHero />
      <OverviewStats />
      <section className="layout-grid xl:grid-cols-[1.2fr_0.8fr]">
        <ActivityFeed />
        <OverviewRail />
      </section>
    </div>
  );
}
