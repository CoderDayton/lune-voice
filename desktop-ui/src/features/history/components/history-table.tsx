// SPDX-License-Identifier: GPL-3.0-or-later

import { plural } from '@lingui/core/macro';
import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { invoke } from '@tauri-apps/api/core';
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  HardDrive,
} from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ConfirmDialog } from '../../../components/feedback/confirm-dialog';
import {
  LIST_HEAD_CELL,
  LIST_ROW,
  ListCard,
  ListEmpty,
  ListFooter,
  ListScroll,
  ListSearchRail,
  useListFilterHotkey,
} from '../../../components/ui/list-card';
import { LIMITS } from '../../../lib/backend/generated/limits';
import { formatDuration, transcriptFileName } from '../../../lib/session-format';
import { formatBackend } from '../../../store/workspace-mappers';
import {
  fetchHistoryPage,
  getHistoryServerPageCount,
  type HistorySession,
  removeHistorySession,
  removeHistorySessions,
  workspaceActions,
  workspaceStore,
} from '../../../store/workspace-store';

/**
 * Sessions table — the whole history surface in one card: integrated search,
 * a sticky-header data table whose rows expand in place to reveal the full
 * transcript and actions, and pinned pagination. Fills its parent's height;
 * only the table body scrolls.
 *
 * Shares its card, search rail, and table chrome with Vocabulary via
 * `components/ui/list-card`, so the two lists stay identical.
 */
export function HistoryTable() {
  const { t } = useLingui();
  const sessions = useStore(workspaceStore, (s) => s.historyPageItems);
  // `visibleCount` = sessions matching the query (server total for this page
  // set); `totalCount` = the lifetime session count (unfiltered), matching the
  // overview. They coincide when the search box is empty.
  const visibleCount = useStore(workspaceStore, (s) => s.historyTotal);
  const totalCount = useStore(workspaceStore, (s) => s.sessionTotals.sessions);
  const loading = useStore(workspaceStore, (s) => s.historyLoading);
  const search = useStore(workspaceStore, (s) => s.historySearch);
  const page = useStore(workspaceStore, (s) => s.historyPage);
  const pageSize = useStore(workspaceStore, (s) => s.historyPageSize);
  const totalPages = useStore(workspaceStore, getHistoryServerPageCount);
  const historyDays = useStore(workspaceStore, (s) => s.settings.sessionHistoryDays);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  // Selected ids, not selected rows: the set survives paging and searching, so
  // a sweep across several pages can be deleted in one go. Ids that have
  // scrolled out of view are still valid — the backend skips any that are
  // already gone.
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set());
  const searchRef = useRef<HTMLInputElement>(null);

  // Load the first page on mount; page/search changes refetch via their store
  // actions (`setHistoryPage` / `setHistorySearch`).
  useEffect(() => {
    void fetchHistoryPage();
  }, []);

  useListFilterHotkey(searchRef);

  const hasSearch = search.trim().length > 0;
  const start = visibleCount === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, visibleCount);

  const pageIds = sessions.map((s) => s.id);
  const selectedOnPage = pageIds.filter((id) => selected.has(id)).length;
  const allOnPageSelected = pageIds.length > 0 && selectedOnPage === pageIds.length;

  function toggleSelected(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  }

  // Select-all covers the current page only — it is the visible list, and a
  // button that silently claimed thousands of unseen rows would be a trap.
  function toggleSelectPage() {
    setSelected((current) => {
      const next = new Set(current);
      for (const id of pageIds) {
        if (allOnPageSelected) next.delete(id);
        else next.add(id);
      }
      return next;
    });
  }

  async function deleteSelected() {
    // Clear only once the delete lands. A failed delete leaves the rows in
    // place, so dropping the selection early would make the user re-tick every
    // box before retrying.
    if (await removeHistorySessions([...selected])) {
      setSelected(new Set());
      setExpandedId(null);
    }
  }

  return (
    <ListCard>
      <ListSearchRail
        inputRef={searchRef}
        value={search}
        onValueChange={(v) => workspaceActions.setHistorySearch(v)}
        placeholder={t`Search transcripts…`}
        label={t`Search transcripts`}
        count={hasSearch ? t`${visibleCount} / ${totalCount}` : t`${totalCount} sessions`}
      />

      {selected.size > 0 && (
        <SelectionBar
          count={selected.size}
          onClear={() => setSelected(new Set())}
          onDelete={() => void deleteSelected()}
        />
      )}

      <ListScroll>
        {sessions.length === 0 ? (
          <ListEmpty>
            {loading ? (
              <span className="loading loading-spinner loading-md text-content-subtle" />
            ) : (
              <p className="text-sm text-content-muted">
                {hasSearch
                  ? t`No sessions match "${search.trim()}".`
                  : totalCount === 0
                    ? t`No sessions yet. Start dictating to see your transcripts here.`
                    : t`No sessions found.`}
              </p>
            )}
          </ListEmpty>
        ) : (
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className={`${LIST_HEAD_CELL} w-px pl-3`}>
                  <input
                    type="checkbox"
                    className="checkbox checkbox-sm align-middle"
                    checked={allOnPageSelected}
                    ref={(el) => {
                      // Partial page selection reads as "some", not "none".
                      if (el) el.indeterminate = selectedOnPage > 0 && !allOnPageSelected;
                    }}
                    onChange={toggleSelectPage}
                    aria-label={t`Select all sessions on this page`}
                  />
                </th>
                <th className={`${LIST_HEAD_CELL} w-px pl-2 pr-1`} />
                <th className={`${LIST_HEAD_CELL} px-2`}>{t`Time`}</th>
                <th className={`${LIST_HEAD_CELL} px-2`}>{t`Transcript`}</th>
                <th className={`${LIST_HEAD_CELL} hidden px-2 text-right sm:table-cell`}>
                  {t`Words`}
                </th>
                <th className={`${LIST_HEAD_CELL} hidden px-2 text-right sm:table-cell`}>
                  {t`Length`}
                </th>
                <th className={`${LIST_HEAD_CELL} hidden px-2 lg:table-cell`}>{t`Engine`}</th>
                <th className={`${LIST_HEAD_CELL} w-px pr-3`} />
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => (
                <SessionRows
                  key={session.id}
                  session={session}
                  expanded={expandedId === session.id}
                  selected={selected.has(session.id)}
                  onSelect={() => toggleSelected(session.id)}
                  onToggle={() =>
                    setExpandedId((current) => (current === session.id ? null : session.id))
                  }
                />
              ))}
            </tbody>
          </table>
        )}
      </ListScroll>

      {/* Always present, like Home's footer: the count and retention note
          answer "is this everything?" whether or not there is a second page. */}
      <ListFooter>
        <div className="flex w-full flex-wrap items-center gap-x-4 gap-y-1.5 text-sm text-content-subtle">
          <span className="tabular-nums">
            {visibleCount === 0
              ? t`No sessions`
              : t`Showing ${start}–${end} of ${visibleCount} sessions`}
          </span>
          <span
            className="inline-block h-3 w-px shrink-0 bg-surface-border-strong"
            aria-hidden="true"
          />
          <span className="tabular-nums">
            {historyDays == null
              ? t`History isn't saved`
              : historyDays === LIMITS.historyDaysForever
                ? t`History kept forever`
                : t`History kept ${historyDays} days`}
          </span>
          <span
            className="inline-block h-3 w-px shrink-0 bg-surface-border-strong"
            aria-hidden="true"
          />
          <span className="inline-flex items-center gap-1.5">
            <HardDrive size={13} strokeWidth={1.7} />
            {t`Stored on this device`}
          </span>
          {totalPages > 1 && (
            <nav aria-label={t`Pagination`} className="ml-auto join">
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
            </nav>
          )}
        </div>
      </ListFooter>
    </ListCard>
  );
}

/**
 * Bulk-action bar, shown only while something is selected.
 *
 * Sits between the search rail and the scroll region rather than floating over
 * the rows, so it never covers the transcripts you are picking from.
 */
function SelectionBar({
  count,
  onClear,
  onDelete,
}: {
  count: number;
  onClear: () => void;
  onDelete: () => void;
}) {
  const { t } = useLingui();
  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-surface-border bg-surface-nav-active px-3 py-2 sm:px-4">
      <span className="text-sm tabular-nums text-content">{t`${count} selected`}</span>
      <div className="ml-auto flex items-center gap-2">
        <button type="button" className="btn btn-ghost btn-sm" onClick={onClear}>
          {t`Clear selection`}
        </button>
        <ConfirmDialog
          title={t`Delete ${plural(count, { one: '# session', other: '# sessions' })}?`}
          description={t`This will permanently remove the selected dictation sessions and their transcripts.`}
          triggerLabel={t`Delete selected`}
          triggerClassName="btn btn-soft btn-error btn-sm gap-1.5"
          onConfirm={onDelete}
        />
      </div>
    </div>
  );
}

interface SessionRowsProps {
  session: HistorySession;
  expanded: boolean;
  selected: boolean;
  onSelect: () => void;
  onToggle: () => void;
}

function SessionRows({ session, expanded, selected, onSelect, onToggle }: SessionRowsProps) {
  const { t } = useLingui();
  const [copied, setCopied] = useState(false);
  // Keeps the detail row mounted through the fold-up animation, so closing can
  // finish animating before the row leaves the DOM.
  const [shown, setShown] = useState(expanded);
  const detailRef = useRef<HTMLTableRowElement>(null);
  const foldRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<Animation | null>(null);
  const followRef = useRef(0);

  // Unfold / fold the detail panel. The row is conditionally rendered, so a CSS
  // transition never sees both states — instead the wrapper's height is
  // animated with WAAPI, starting from whatever height an in-flight animation
  // left it at, so toggling mid-motion reverses smoothly instead of jumping.
  // While unfolding, a per-frame `scrollIntoView({ block: 'nearest' })` keeps
  // the growing panel in view — rows near the bottom of the scroll region
  // scroll in step with the unfold instead of landing off-screen (`nearest` is
  // position-aware: it does nothing once the panel is fully visible).
  useLayoutEffect(() => {
    if (expanded && !shown) {
      setShown(true); // mount first; the next pass runs the unfold
      return;
    }
    const el = foldRef.current;
    if (!shown || !el) return;

    const midFlight = animRef.current?.playState === 'running';
    const fromHeight = midFlight
      ? el.getBoundingClientRect().height
      : expanded
        ? 0
        : el.offsetHeight;
    const fromOpacity = midFlight ? Number(getComputedStyle(el).opacity) : expanded ? 0 : 1;
    animRef.current?.cancel();
    cancelAnimationFrame(followRef.current);

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const root = getComputedStyle(document.documentElement);

    if (expanded) {
      const anim = el.animate(
        [
          { height: `${fromHeight}px`, opacity: fromOpacity },
          { height: `${el.offsetHeight}px`, opacity: 1 },
        ],
        {
          duration: reduced ? 0 : 260,
          easing: root.getPropertyValue('--ease-decelerate').trim() || 'ease-out',
        },
      );
      animRef.current = anim;
      const follow = () => {
        detailRef.current?.scrollIntoView({ block: 'nearest' });
        if (anim.playState === 'running') followRef.current = requestAnimationFrame(follow);
      };
      follow();
    } else {
      const anim = el.animate(
        [
          { height: `${fromHeight}px`, opacity: fromOpacity },
          { height: '0px', opacity: 0 },
        ],
        {
          duration: reduced ? 0 : 190,
          easing: root.getPropertyValue('--ease-accelerate').trim() || 'ease-in',
          fill: 'forwards',
        },
      );
      animRef.current = anim;
      anim.onfinish = () => setShown(false);
    }

    return () => cancelAnimationFrame(followRef.current);
  }, [expanded, shown]);

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
    try {
      // The save dialog runs in Rust: the chosen path never passes through
      // here, so this side can only suggest a name, never a destination.
      await invoke('export_transcript', {
        transcript: session.transcript,
        defaultName: transcriptFileName(session.startedAt),
        title: t`Save Transcript`,
      });
    } catch (e) {
      console.warn('[transcript-download]', e);
    }
  }

  return (
    <>
      <tr
        onClick={onToggle}
        aria-expanded={expanded}
        className={`${LIST_ROW} cursor-pointer align-top ${
          expanded || selected ? 'bg-surface-nav-active' : ''
        }`}
      >
        <td className="py-3 pl-3">
          {/* `h-5` is the first text line's line-height, so both controls centre
              on the time rather than hanging off the top of an `align-top` cell. */}
          <div className="flex h-5 items-center">
            <input
              type="checkbox"
              className="checkbox checkbox-sm"
              checked={selected}
              onChange={onSelect}
              // The row itself toggles the transcript fold; ticking the box must
              // not also expand the row it belongs to.
              onClick={(e) => e.stopPropagation()}
              aria-label={t`Select session`}
            />
          </div>
        </td>
        <td className="py-3 pl-2 pr-1">
          <div className="flex h-5 items-center">
            <ChevronDown
              size={15}
              className={`text-content-subtle transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
            />
          </div>
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
      {shown && (
        <tr ref={detailRef} className="border-b border-surface-border bg-surface-inset/40">
          <td colSpan={8} className="p-0">
            <div ref={foldRef} className="overflow-hidden">
              <div className="px-4 py-4 sm:px-6">
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
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
