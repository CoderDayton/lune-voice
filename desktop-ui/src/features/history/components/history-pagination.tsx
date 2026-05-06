import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import {
  getHistoryPageCount,
  getVisibleSessionCount,
  workspaceActions,
  workspaceStore,
} from '../../../store/workspace-store';

export function HistoryPagination() {
  const { t } = useLingui();
  const page = useStore(workspaceStore, (s) => s.historyPage);
  const pageSize = useStore(workspaceStore, (s) => s.historyPageSize);
  const totalPages = useStore(workspaceStore, getHistoryPageCount);
  const visibleCount = useStore(workspaceStore, getVisibleSessionCount);

  if (totalPages <= 1) return null;

  const start = (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, visibleCount);

  return (
    <nav
      aria-label={t`Pagination`}
      className="flex items-center justify-between gap-3 sm:px-1 mt-2"
    >
      <span className="text-sm text-content-subtle tabular-nums">
        {t`${start}–${end} of ${visibleCount}`}
      </span>
      <div className="join">
        <button
          type="button"
          className="btn btn-sm join-item gap-1"
          disabled={page <= 1}
          onClick={() => workspaceActions.setHistoryPage(page - 1)}
          aria-label={t`Previous page`}
        >
          <ChevronLeft size={14} /> {t`Prev`}
        </button>
        <span
          className="btn btn-sm join-item btn-ghost no-animation pointer-events-none tabular-nums"
          aria-current="page"
        >
          {t`${page} / ${totalPages}`}
        </span>
        <button
          type="button"
          className="btn btn-sm join-item gap-1"
          disabled={page >= totalPages}
          onClick={() => workspaceActions.setHistoryPage(page + 1)}
          aria-label={t`Next page`}
        >
          {t`Next`} <ChevronRight size={14} />
        </button>
      </div>
    </nav>
  );
}
