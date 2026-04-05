import { memo } from "react";
import { useStore } from "@tanstack/react-store";
import { useLingui } from "@lingui/react/macro";
import {
  getVisibleHistorySessions,
  workspaceStore,
} from "../../../store/workspace-store";
import type { HistorySession } from "../../../store/workspace-store";
import { SessionDetailModal } from "./session-detail-modal";

interface SessionRowProps {
  item: HistorySession;
}

const SessionRow = memo(function SessionRow({ item }: SessionRowProps) {
  const { t } = useLingui();
  const time = new Date(item.startedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

  return (
    <SessionDetailModal session={item}>
      {(open) => (
        <li className="list-row cursor-pointer" onClick={open}>
          <div className="list-col-grow min-w-0">
            <p className="text-sm line-clamp-2 sm:line-clamp-3">{item.transcript || t`Empty session`}</p>
            <div className="text-xs text-content-subtle mt-0.5">
              <span className="tabular-nums">{time}</span>
              <span className="mx-1">·</span>
              {item.words} {t`words`}
              <span className="mx-1">·</span>
              {item.seconds}{t`s`}
            </div>
          </div>
        </li>
      )}
    </SessionDetailModal>
  );
});

export function HistorySessionList() {
  const { t } = useLingui();
  const visibleSessions = useStore(workspaceStore, getVisibleHistorySessions);
  const allSessions = useStore(workspaceStore, (s) => s.historySessions);
  const searchQuery = useStore(workspaceStore, (s) => s.historySearch);

  const groupMap = new Map<string, typeof visibleSessions>();
  for (const session of visibleSessions) {
    const label = formatGroupLabel(session.startedAt);
    const items = groupMap.get(label);
    if (items) items.push(session);
    else groupMap.set(label, [session]);
  }
  const groups = Array.from(groupMap.entries()).map(([date, items]) => ({ date, items }));

  if (groups.length === 0) {
    const hasSearch = searchQuery.trim().length > 0;
    return (
      <div className="alert">
        <span>
          {hasSearch
            ? t`No sessions match "${searchQuery.trim()}". Try a different search.`
            : allSessions.length === 0
              ? t`No sessions yet. Start dictating to see your transcripts here.`
              : t`No sessions found.`}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {groups.map((group) => (
        <div key={group.date}>
          <h3 className="sticky top-0 z-10 text-xs font-semibold uppercase tracking-wider text-content-subtle mb-2 px-1 py-1 backdrop-blur-sm">
            {group.date}
          </h3>
          <ul className="list rounded-box border border-surface-border bg-surface-card">
            {group.items.map((item) => (
              <SessionRow key={item.id} item={item} />
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
