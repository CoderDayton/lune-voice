import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { memo } from 'react';
import type { HistorySession } from '../../../store/workspace-store';
import {
  getPaginatedHistorySessions,
  getVisibleSessionCount,
  workspaceStore,
} from '../../../store/workspace-store';
import { SessionDetailModal } from './session-detail-modal';

interface HistorySessionListProps {
  /** When provided, rows select into a detail pane instead of opening the
   *  modal (desktop master-detail mode). */
  selectedId?: string | null;
  onSelect?: (session: HistorySession) => void;
}

function SessionRowBody({ item }: { item: HistorySession }) {
  const { t } = useLingui();
  const time = new Date(item.startedAt).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });
  return (
    <div className="list-col-grow min-w-0">
      <p className="line-clamp-2 text-[0.95rem] leading-6 text-content">
        {item.transcript || t`Empty session`}
      </p>
      <div className="mt-1 flex items-center gap-1.5 text-xs text-content-subtle">
        <span className="tabular-nums">{time}</span>
        <span aria-hidden="true">·</span>
        {item.words} {t`words`}
        <span aria-hidden="true">·</span>
        {item.seconds}
        {t`s`}
      </div>
    </div>
  );
}

interface SessionRowProps {
  item: HistorySession;
  selected?: boolean;
  onSelect?: (session: HistorySession) => void;
}

const SessionRow = memo(function SessionRow({ item, selected, onSelect }: SessionRowProps) {
  const { t } = useLingui();

  // Master-detail mode: the row selects into the side pane.
  if (onSelect) {
    return (
      <li className="list-row p-0">
        <button
          type="button"
          aria-current={selected ? 'true' : undefined}
          onClick={() => onSelect(item)}
          className={`interactive-row flex min-h-20 w-full items-center rounded-none px-4 py-3 text-left focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-primary ${
            selected
              ? 'border-l-2 border-primary bg-surface-nav-active pl-[0.875rem] text-content-strong'
              : ''
          }`}
        >
          <SessionRowBody item={item} />
        </button>
      </li>
    );
  }

  // Default mode: the row opens the modal.
  return (
    <SessionDetailModal session={item}>
      {(open) => (
        <li className="list-row p-0">
          <button
            type="button"
            className="interactive-row flex min-h-20 w-full items-center rounded-none px-4 py-3 text-left focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-primary"
            aria-haspopup="dialog"
            aria-label={t`Open session details`}
            onClick={open}
          >
            <SessionRowBody item={item} />
          </button>
        </li>
      )}
    </SessionDetailModal>
  );
});

export function HistorySessionList({ selectedId, onSelect }: HistorySessionListProps = {}) {
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
    <div className="flex flex-col gap-2 sm:gap-3">
      {groups.map((group) => (
        <div key={group.date}>
          <h3 className="mb-2 px-1 text-xs font-semibold uppercase tracking-[0.16em] text-content-subtle">
            {group.date}
          </h3>
          <ul className="panel-surface divide-y divide-surface-border overflow-hidden">
            {group.items.map((item) => (
              <SessionRow
                key={item.id}
                item={item}
                selected={selectedId === item.id}
                onSelect={onSelect}
              />
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
