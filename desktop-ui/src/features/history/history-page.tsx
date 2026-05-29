import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Inbox } from 'lucide-react';
import { useEffect, useState } from 'react';
import { ConfirmDialog } from '../../components/feedback/confirm-dialog';
import {
  getPaginatedHistorySessions,
  removeHistorySession,
  workspaceStore,
} from '../../store/workspace-store';
import { HistorySessionList } from './components/history-session-list';
import { HistoryToolbar } from './components/history-toolbar';
import { SessionDetailContent } from './components/session-detail-modal';
import { useMediaQuery } from './use-media-query';

export function HistoryPage() {
  const { t } = useLingui();
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);
  const isDesktop = useMediaQuery('(min-width: 1024px)');
  const visible = useStore(workspaceStore, getPaginatedHistorySessions);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Keep a valid selection on desktop: default to the first visible session,
  // and re-point if the current one scrolls out of the page (e.g. pagination,
  // search, or deletion).
  useEffect(() => {
    if (!isDesktop) return;
    if (visible.length === 0) {
      setSelectedId(null);
      return;
    }
    setSelectedId((current) =>
      current && visible.some((s) => s.id === current) ? current : visible[0].id,
    );
  }, [isDesktop, visible]);

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

  // Mobile / tablet: stacked list, detail opens in the modal.
  if (!isDesktop) {
    return (
      <div className="page-stack">
        <HistoryToolbar />
        <HistorySessionList />
      </div>
    );
  }

  // Desktop: master-detail. The list selects into a sticky detail pane.
  const selected = visible.find((s) => s.id === selectedId) ?? null;

  return (
    <div className="flex flex-col gap-space-md">
      <HistoryToolbar />
      <div className="bento-board">
        <div className="col-span-5 min-w-0">
          <HistorySessionList
            selectedId={selectedId}
            onSelect={(session) => setSelectedId(session.id)}
          />
        </div>
        <div className="col-span-7 min-w-0">
          <div className="bento-cell sticky top-4 overflow-hidden">
            {selected ? (
              <>
                <SessionDetailContent session={selected} />
                <div className="flex justify-end border-t border-surface-border px-5 py-3 sm:px-6">
                  <ConfirmDialog
                    title={t`Delete session?`}
                    description={t`This will permanently remove this dictation session and its transcript.`}
                    triggerLabel={t`Delete`}
                    onConfirm={() => removeHistorySession(selected.id)}
                  />
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center px-6 py-20 text-center">
                <div className="token-icon mb-4 size-14 rounded-full">
                  <Inbox className="size-7" strokeWidth={1.3} />
                </div>
                <p className="text-sm text-content-muted">
                  {t`Select a session to view its transcript`}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
