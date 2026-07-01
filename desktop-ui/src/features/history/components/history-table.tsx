import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Search,
  X,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { ConfirmDialog } from '../../../components/feedback/confirm-dialog';
import {
  getHistoryPageCount,
  getPaginatedHistorySessions,
  getVisibleSessionCount,
  type HistorySession,
  removeHistorySession,
  workspaceActions,
  workspaceStore,
} from '../../../store/workspace-store';

function formatBackend(backend: string): string {
  if (backend.includes('WhisperCpp')) return 'Whisper';
  if (backend.includes('OpenAi')) return 'Cloud';
  return backend;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/**
 * Sessions table — the whole history surface in one card: integrated search,
 * a sticky-header data table whose rows expand in place to reveal the full
 * transcript and actions, and pinned pagination. Fills its parent's height;
 * only the table body scrolls.
 */
export function HistoryTable() {
  const { t } = useLingui();
  const sessions = useStore(workspaceStore, getPaginatedHistorySessions);
  const visibleCount = useStore(workspaceStore, getVisibleSessionCount);
  const totalCount = useStore(workspaceStore, (s) => s.historySessions.length);
  const search = useStore(workspaceStore, (s) => s.historySearch);
  const page = useStore(workspaceStore, (s) => s.historyPage);
  const pageSize = useStore(workspaceStore, (s) => s.historyPageSize);
  const totalPages = useStore(workspaceStore, getHistoryPageCount);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // "/" focuses the search field, matching the rest of the app.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target?.isContentEditable ?? false);
      if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const hasSearch = search.trim().length > 0;
  const start = visibleCount === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, visibleCount);

  return (
    <div className="panel-surface flex h-full min-h-0 flex-col overflow-hidden">
      {/* Search row */}
      <div className="flex shrink-0 items-center gap-3 border-b border-surface-border px-3 py-2.5 sm:px-4">
        <Search size={17} strokeWidth={2} className="shrink-0 text-content-subtle" />
        <input
          ref={searchRef}
          type="search"
          value={search}
          onChange={(e) => workspaceActions.setHistorySearch(e.target.value)}
          placeholder={t`Search transcripts…`}
          className="h-9 min-w-0 grow bg-transparent text-sm text-content placeholder:text-content-subtle"
          aria-label={t`Search transcripts`}
        />
        {hasSearch ? (
          <button
            type="button"
            onClick={() => workspaceActions.setHistorySearch('')}
            className="btn btn-soft btn-circle btn-xs"
            aria-label={t`Clear search`}
          >
            <X size={14} />
          </button>
        ) : (
          <kbd className="kbd kbd-sm hidden text-content-subtle sm:inline-flex">/</kbd>
        )}
        <span className="hidden shrink-0 text-sm tabular-nums text-content-subtle sm:inline">
          {hasSearch ? t`${visibleCount} / ${totalCount}` : t`${totalCount} sessions`}
        </span>
      </div>

      {/* Table body — the only scroll region */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center px-6 py-16 text-center">
            <p className="text-sm text-content-muted">
              {hasSearch
                ? t`No sessions match "${search.trim()}".`
                : totalCount === 0
                  ? t`No sessions yet. Start dictating to see your transcripts here.`
                  : t`No sessions found.`}
            </p>
          </div>
        ) : (
          <table className="w-full border-collapse text-sm">
            <thead className="sticky top-0 z-10 bg-surface-panel">
              <tr className="border-b border-surface-border text-left text-xs uppercase tracking-wider text-content-subtle">
                <th className="w-px py-2 pl-3" />
                <th className="px-2 py-2 font-medium">{t`Time`}</th>
                <th className="px-2 py-2 font-medium">{t`Transcript`}</th>
                <th className="hidden px-2 py-2 text-right font-medium sm:table-cell">{t`Words`}</th>
                <th className="hidden px-2 py-2 text-right font-medium sm:table-cell">{t`Length`}</th>
                <th className="hidden px-2 py-2 font-medium lg:table-cell">{t`Engine`}</th>
                <th className="w-px py-2 pr-3" />
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => (
                <SessionRows
                  key={session.id}
                  session={session}
                  expanded={expandedId === session.id}
                  onToggle={() =>
                    setExpandedId((current) => (current === session.id ? null : session.id))
                  }
                />
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <nav
          aria-label={t`Pagination`}
          className="flex shrink-0 items-center justify-between gap-3 border-t border-surface-border px-3 py-2.5 sm:px-4"
        >
          <span className="text-sm tabular-nums text-content-subtle">
            {t`${start}–${end} of ${visibleCount}`}
          </span>
          <div className="join">
            <button
              type="button"
              className="btn btn-soft btn-sm join-item gap-1"
              disabled={page <= 1}
              onClick={() => workspaceActions.setHistoryPage(page - 1)}
              aria-label={t`Previous page`}
            >
              <ChevronLeft size={14} /> {t`Prev`}
            </button>
            <span
              className="btn btn-soft btn-sm join-item pointer-events-none no-animation tabular-nums"
              aria-current="page"
            >
              {t`${page} / ${totalPages}`}
            </span>
            <button
              type="button"
              className="btn btn-soft btn-sm join-item gap-1"
              disabled={page >= totalPages}
              onClick={() => workspaceActions.setHistoryPage(page + 1)}
              aria-label={t`Next page`}
            >
              {t`Next`} <ChevronRight size={14} />
            </button>
          </div>
        </nav>
      )}
    </div>
  );
}

interface SessionRowsProps {
  session: HistorySession;
  expanded: boolean;
  onToggle: () => void;
}

function SessionRows({ session, expanded, onToggle }: SessionRowsProps) {
  const { t } = useLingui();
  const [copied, setCopied] = useState(false);
  const detailRef = useRef<HTMLTableRowElement>(null);

  // When a row expands, the detail panel unfolds *below* the summary row. For
  // rows near the bottom of the scroll region that lands off-screen, so pull it
  // into view — `nearest` is position-aware: it does nothing when already
  // visible and scrolls the minimum needed otherwise (no jump for top rows).
  useEffect(() => {
    if (expanded) detailRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [expanded]);

  const startDate = new Date(session.startedAt);
  const time = startDate.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  const date = startDate.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

  function copy() {
    if (!session.transcript) return;
    navigator.clipboard
      .writeText(session.transcript)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {});
  }

  async function download() {
    if (!session.transcript) return;
    const d = new Date(session.startedAt);
    const stamp = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}-${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
    try {
      const path = await save({
        title: t`Save Transcript`,
        defaultPath: `lune-${stamp}.txt`,
        filters: [{ name: 'Text', extensions: ['txt'] }],
      });
      if (path) await writeTextFile(path, session.transcript);
    } catch (e) {
      console.warn('[transcript-download]', e);
    }
  }

  return (
    <>
      <tr
        onClick={onToggle}
        aria-expanded={expanded}
        className={`cursor-pointer border-b border-surface-border/60 align-top transition-colors hover:bg-surface-hover ${
          expanded ? 'bg-surface-nav-active' : ''
        }`}
      >
        <td className="py-3 pl-3">
          <ChevronDown
            size={15}
            className={`text-content-subtle transition-transform ${expanded ? 'rotate-180' : ''}`}
          />
        </td>
        <td className="whitespace-nowrap px-2 py-3 tabular-nums">
          <div className="text-content">{time}</div>
          <div className="text-xs text-content-subtle">{date}</div>
        </td>
        <td className="px-2 py-3 text-content">
          <p className={expanded ? '' : 'line-clamp-1 sm:line-clamp-2'}>
            {session.transcript || t`Empty session`}
          </p>
        </td>
        <td className="hidden px-2 py-3 text-right tabular-nums text-content-muted sm:table-cell">
          {session.words}
        </td>
        <td className="hidden px-2 py-3 text-right tabular-nums text-content-muted sm:table-cell">
          {formatDuration(session.seconds)}
        </td>
        <td className="hidden px-2 py-3 text-content-muted lg:table-cell">
          {formatBackend(session.backend)}
        </td>
        <td className="py-3 pr-3" />
      </tr>
      {expanded && (
        <tr ref={detailRef} className="border-b border-surface-border bg-surface-inset/40">
          <td colSpan={7} className="px-4 py-4 sm:px-6">
            <div className="rounded-field border border-surface-border bg-surface-inset p-4">
              <p className="max-h-60 overflow-y-auto whitespace-pre-wrap break-words text-[0.95rem] leading-7 text-content">
                {session.transcript || t`No transcript`}
              </p>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs tabular-nums text-content-subtle">
              <span>
                {session.words} {t`words`}
              </span>
              <span aria-hidden="true">·</span>
              <span>{formatDuration(session.seconds)}</span>
              <span aria-hidden="true">·</span>
              <span>{formatBackend(session.backend)}</span>
              <span aria-hidden="true">·</span>
              <span className="capitalize">{session.style || t`None`}</span>
              <span aria-hidden="true">·</span>
              <span>{session.status}</span>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button type="button" onClick={copy} className="btn btn-soft btn-sm gap-1.5">
                {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                {copied ? t`Copied` : t`Copy`}
              </button>
              <button type="button" onClick={download} className="btn btn-soft btn-sm gap-1.5">
                <Download size={14} /> {t`Download`}
              </button>
              <div className="ml-auto">
                <ConfirmDialog
                  title={t`Delete session?`}
                  description={t`This will permanently remove this dictation session and its transcript.`}
                  triggerLabel={t`Delete`}
                  onConfirm={() => removeHistorySession(session.id)}
                />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
