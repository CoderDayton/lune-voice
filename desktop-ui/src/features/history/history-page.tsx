import { useStore } from '@tanstack/react-store';
import { workspaceStore } from '../../store/workspace-store';
import { HistorySessionList } from './components/history-session-list';
import { HistoryToolbar } from './components/history-toolbar';

export function HistoryPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="page-stack">
        <div className="panel-surface flex flex-col gap-3 p-3 sm:flex-row sm:items-center sm:gap-3">
          <div className="skeleton-shimmer h-12 flex-1 rounded-field" />
          <div className="skeleton-shimmer h-4 w-24 shrink-0" />
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
      <HistoryToolbar />
      <HistorySessionList />
    </div>
  );
}
