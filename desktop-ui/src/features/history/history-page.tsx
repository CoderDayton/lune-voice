import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { getVisibleSessionCount, workspaceStore } from '../../store/workspace-store';
import { HistorySearch } from './components/history-search';
import { HistorySessionList } from './components/history-session-list';

export function HistoryPage() {
  const { t } = useLingui();
  const count = useStore(workspaceStore, getVisibleSessionCount);
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="page-stack">
        <div className="toolbar-surface flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
          <div className="flex-1">
            <div className="skeleton-shimmer h-9 w-full" />
          </div>
          <div className="skeleton-shimmer h-4 w-20 shrink-0" />
        </div>
        <div className="flex flex-col gap-2">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="panel-surface flex flex-col gap-2 p-4">
              <div className="flex justify-between items-center">
                <div className="skeleton-shimmer h-4 w-48" />
                <div className="skeleton-shimmer h-3 w-16" />
              </div>
              <div className="skeleton-shimmer h-3 w-full" />
              <div className="skeleton-shimmer h-3 w-3/4" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <div className="toolbar-surface flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
        <div className="flex-1">
          <HistorySearch />
        </div>
        <span className="text-sm text-content-subtle tabular-nums shrink-0">
          {count} {t`sessions`}
        </span>
      </div>
      <HistorySessionList />
    </div>
  );
}
