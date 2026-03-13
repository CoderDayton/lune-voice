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
    <Panel className="p-4 sm:p-6">
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

      <div className="flex flex-col gap-4 mt-6">
        {recentActivity.length === 0 ? (
          <p className="body-copy-soft text-center py-8">No sessions yet. Start dictating to see recent activity.</p>
        ) : (
          recentActivity.map((item) => (
            <article key={item.id} className="card bg-base-300/50 border border-base-300/30 p-3 sm:p-4 stack-copy">
              <div className="flex flex-wrap gap-2">
                {item.meta.map((entry) => (
                  <Pill key={entry}>{entry}</Pill>
                ))}
              </div>
              <h4 className="font-semibold mt-4">{item.title}</h4>
              <p className="body-copy-strong mt-2 line-clamp-3">{item.text}</p>
            </article>
          ))
        )}
      </div>
    </Panel>
  );
}
