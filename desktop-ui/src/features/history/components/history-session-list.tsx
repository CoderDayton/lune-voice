import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import {
  formatBackendLabel,
  getVisibleHistorySessions,
  workspaceStore,
} from "../../../store/workspace-store";

export function HistorySessionList() {
  const visibleSessions = useStore(workspaceStore, getVisibleHistorySessions);

  const filteredGroups = useMemo(() => {
    const groups = new Map<string, typeof visibleSessions>();

    for (const session of visibleSessions) {
      const label = formatGroupLabel(session.startedAt);
      const items = groups.get(label);

      if (items) {
        items.push(session);
      } else {
        groups.set(label, [session]);
      }
    }

    return Array.from(groups.entries()).map(([date, items]) => ({ date, items }));
  }, [visibleSessions]);

  return (
    <Panel className="overflow-hidden p-0">
      {filteredGroups.map((group) => (
        <section key={group.date}>
          <div className="session-group-head">
            <span className="label-text-custom">{group.date}</span>
          </div>

          <div>
            {group.items.map((item) => (
              <article key={item.id} className="session-row">
                <div className="flex flex-wrap gap-2">
                  <Pill>{formatTime(item.startedAt)}</Pill>
                  <Pill>{item.words} words</Pill>
                  <Pill>{item.seconds}s audio</Pill>
                  <Pill>{formatBackendLabel(item.backend)}</Pill>
                </div>
                <p className="body-copy-strong max-w-[56rem] mt-4">{item.transcript}</p>
              </article>
            ))}
          </div>
        </section>
      ))}
    </Panel>
  );
}

function formatGroupLabel(value: string) {
  const date = new Date(value);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);

  if (date.toDateString() === today.toDateString()) {
    return "Today";
  }

  if (date.toDateString() === yesterday.toDateString()) {
    return "Yesterday";
  }

  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}
