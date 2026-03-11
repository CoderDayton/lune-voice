import { useStore } from "@tanstack/react-store";
import { Link } from "@tanstack/react-router";
import { buttonClassName } from "../../../components/ui/button";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { formatBackendLabel, workspaceStore } from "../../../store/workspace-store";

export function ActivityFeed() {
  const recentActivity = useStore(workspaceStore, (state) =>
    state.historySessions.slice(0, 3).map((session) => ({
      id: session.id,
      title: session.transcript.split(".")[0] || "Recent session",
      text: session.transcript,
      meta: [
        new Date(session.startedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
        `${session.words} words`,
        `${session.seconds}s`,
        formatBackendLabel(session.backend),
      ],
    })),
  );

  return (
    <Panel className="panel-pad-lg">
      <div className="header-row">
        <div>
          <h3 className="section-title">Recent activity</h3>
          <p className="body-copy-soft mt-2">
            Latest transcriptions from this device, with enough context to judge quality fast.
          </p>
        </div>
        <Link to="/history" className={buttonClassName("ghost")}>
          View full history
        </Link>
      </div>

      <div className="activity-stack">
        {recentActivity.map((item) => (
          <article key={item.id} className="activity-card">
            <div className="flex flex-wrap gap-2">
              {item.meta.map((entry) => (
                <Pill key={entry}>{entry}</Pill>
              ))}
            </div>
            <h4 className="activity-title mt-4">{item.title}</h4>
            <p className="transcript-copy mt-2">{item.text}</p>
            <div className="action-link mt-4">Add to dictionary</div>
          </article>
        ))}
      </div>
    </Panel>
  );
}
