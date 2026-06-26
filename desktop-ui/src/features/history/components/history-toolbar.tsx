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
    <div className="panel-surface flex flex-col gap-2 p-2 sm:flex-row sm:items-center sm:gap-3">
      <label className="flex h-10 flex-1 items-center gap-3 rounded-field border border-surface-border bg-surface-field px-3.5 transition-colors focus-within:border-surface-border-strong focus-within:outline-2 focus-within:outline-offset-[-2px] focus-within:outline-primary/60">
        <Search size={17} strokeWidth={2} className="shrink-0 text-content-subtle" />
        <input
          ref={inputRef}
          type="search"
          value={search}
          onChange={(e) => workspaceActions.setHistorySearch(e.target.value)}
          placeholder={t`Search transcripts…`}
          className="h-full min-w-0 grow bg-transparent text-sm text-content outline-none placeholder:text-content-subtle"
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
          <kbd className="kbd kbd-sm hidden sm:inline-flex text-content-subtle">/</kbd>
        )}
      </label>
      <div className="flex items-center justify-between gap-2 sm:justify-end">
        <span className="flex h-10 shrink-0 items-center rounded-field border border-surface-border bg-surface-panel px-4 text-sm tabular-nums text-content-subtle">
          {countLabel}
        </span>
      </div>
    </div>
  );
}
