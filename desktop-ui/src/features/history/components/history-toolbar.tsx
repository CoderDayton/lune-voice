import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Search, X } from 'lucide-react';
import { useEffect, useRef } from 'react';
import {
  getVisibleSessionCount,
  workspaceActions,
  workspaceStore,
} from '../../../store/workspace-store';

export function HistoryToolbar() {
  const { t } = useLingui();
  const search = useStore(workspaceStore, (s) => s.historySearch);
  const visibleCount = useStore(workspaceStore, getVisibleSessionCount);
  const totalCount = useStore(workspaceStore, (s) => s.historySessions.length);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTyping =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target?.isContentEditable ?? false);
      if (e.key === '/' && !isTyping && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const filtered = search.trim().length > 0;
  const countLabel = filtered
    ? t`${visibleCount} of ${totalCount} sessions`
    : t`${totalCount} sessions`;

  return (
    <div className="panel-surface flex flex-col gap-3 p-3 sm:flex-row sm:items-center sm:gap-3 sm:p-3">
      <label className="input input-lg flex-1 gap-3 bg-surface-field">
        <Search size={18} strokeWidth={2} className="text-content-subtle shrink-0" />
        <input
          ref={inputRef}
          type="search"
          value={search}
          onChange={(e) => workspaceActions.setHistorySearch(e.target.value)}
          placeholder={t`Search transcripts…`}
          className="grow text-base"
          aria-label={t`Search transcripts`}
        />
        {filtered ? (
          <button
            type="button"
            onClick={() => workspaceActions.setHistorySearch('')}
            className="btn btn-soft btn-circle btn-xs"
            aria-label={t`Clear search`}
          >
            <X size={14} />
          </button>
        ) : (
          <kbd className="kbd kbd-sm hidden sm:inline-flex text-content-dim">/</kbd>
        )}
      </label>
      <div className="flex items-center justify-between gap-2 sm:justify-end">
        <span className="text-sm text-content-subtle tabular-nums shrink-0 px-1">{countLabel}</span>
      </div>
    </div>
  );
}
