import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { memo } from 'react';
import type { HistorySession } from '../../../store/workspace-store';
import {
  getPaginatedHistorySessions,
  getVisibleSessionCount,
  workspaceStore,
} from '../../../store/workspace-store';
import { HistoryPagination } from './history-pagination';
import { SessionDetailModal } from './session-detail-modal';

interface SessionRowProps {
  item: HistorySession;
}

const SessionRow = memo(function SessionRow({ item }: SessionRowProps) {
  const { t } = useLingui();
  const time = new Date(item.startedAt).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });

  return (
    <SessionDetailModal session={item}>
      {(open) => (
        <li className="list-row p-0">
          <button
            type="button"
            className="interactive-row flex w-full items-center text-left px-4 py-3 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-primary"
            aria-haspopup="dialog"
            aria-label={t`Open session details`}
            onClick={open}
          >
            <div className="list-col-grow min-w-0">
              <p className="text-base line-clamp-2 sm:line-clamp-3">
                {item.transcript || t`Empty session`}
              </p>
              <div className="text-sm text-content-subtle mt-0.5">
                <span className="tabular-nums">{time}</span>
                <span className="mx-1">·</span>
                {item.words} {t`words`}
                <span className="mx-1">·</span>
                {item.seconds}
                {t`s`}
              </div>
            </div>
          </button>
        </li>
      )}
    </SessionDetailModal>
  );
});

export function HistorySessionList() {
  const { t } = useLingui();
  const visibleSessions = useStore(workspaceStore, getPaginatedHistorySessions);
  const totalVisible = useStore(workspaceStore, getVisibleSessionCount);
  const totalCount = useStore(workspaceStore, (s) => s.historySessions.length);
  const searchQuery = useStore(workspaceStore, (s) => s.historySearch);

  const today = new Date();
  const todayStr = today.toDateString();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  const yesterdayStr = yesterday.toDateString();

  const groupMap = new Map<string, HistorySession[]>();
  for (const session of visibleSessions) {
    const dateStr = new Date(session.startedAt).toDateString();
    let label: string;
    if (dateStr === todayStr) label = 'Today';
    else if (dateStr === yesterdayStr) label = 'Yesterday';
    else
      label = new Date(session.startedAt).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
      });

    const items = groupMap.get(label);
    if (items) items.push(session);
    else groupMap.set(label, [session]);
  }
  const groups = Array.from(groupMap.entries()).map(([date, items]) => ({ date, items }));

  if (groups.length === 0 && totalVisible === 0) {
    const hasSearch = searchQuery.trim().length > 0;
    return (
      <div role="alert" className="alert alert-info alert-soft panel-surface">
        <span>
          {hasSearch
            ? t`No sessions match "${searchQuery.trim()}". Try a different search.`
            : totalCount === 0
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
          <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-content-subtle mb-2 px-1">
            {group.date}
          </h3>
          <ul className="list panel-surface overflow-hidden">
            {group.items.map((item) => (
              <SessionRow key={item.id} item={item} />
            ))}
          </ul>
        </div>
      ))}
      <HistoryPagination />
    </div>
  );
}
