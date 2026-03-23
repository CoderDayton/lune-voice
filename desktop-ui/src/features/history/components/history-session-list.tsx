import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { useLingui } from "@lingui/react/macro";
import {
  getVisibleHistorySessions,
  workspaceStore,
} from "../../../store/workspace-store";

export function HistorySessionList() {
  const { t } = useLingui();
  const visibleSessions = useStore(workspaceStore, getVisibleHistorySessions);

  const groups = useMemo(() => {
    const map = new Map<string, typeof visibleSessions>();
    for (const session of visibleSessions) {
      const label = formatGroupLabel(session.startedAt);
      const items = map.get(label);
      if (items) items.push(session);
      else map.set(label, [session]);
    }
    return Array.from(map.entries()).map(([date, items]) => ({ date, items }));
  }, [visibleSessions]);

  if (groups.length === 0) {
    return (
      <div className="alert">
        <span>{t`No sessions found. Start dictating to see your transcripts here.`}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {groups.map((group) => (
        <div key={group.date}>
          <h3 className="text-xs font-semibold uppercase tracking-wider opacity-50 mb-2 px-1">
            {group.date}
          </h3>
          <ul className="list bg-base-200 rounded-box">
            {group.items.map((item) => (
              <li key={item.id} className="list-row">
                <div className="text-xs tabular-nums opacity-40">
                  {formatTime(item.startedAt)}
                </div>
                <div>
                  <p className="text-sm line-clamp-2 sm:line-clamp-3">{item.transcript}</p>
                  <div className="flex flex-wrap gap-2 mt-1">
                    <span className="badge badge-sm badge-ghost">{item.words} {t`words`}</span>
                    <span className="badge badge-sm badge-ghost">{item.seconds}{t`s`}</span>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function formatGroupLabel(value: string) {
  const date = new Date(value);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  if (date.toDateString() === today.toDateString()) return "Today";
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}
