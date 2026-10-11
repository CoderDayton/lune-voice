// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Search, X } from 'lucide-react';
import { type ReactNode, type Ref, type RefObject, useEffect } from 'react';

/**
 * Shared chrome for the app's full-height list surfaces (Sessions, Vocabulary).
 *
 * One card owns the pane height and splits it three ways: a search rail on top,
 * exactly one scroll region in the middle, and a footer pinned to the bottom
 * edge. Both pages compose the same parts rather than each restating the
 * layout, so the two lists can't drift apart the way they had.
 *
 * The card fills a flex parent — the page must be marked `fill` in
 * `navigation.ts` for the height to resolve.
 *
 * It draws no frame of its own. It fills the shell's content region, which is
 * already a bordered raised slab, and a card inside that card read as a double
 * edge with a dead gutter between the two. A panel earns a frame here only
 * when it is one of several objects sharing the region — a composer above a
 * list, say — not when it *is* the page.
 */
export function ListCard({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">{children}</div>;
}

/**
 * Focus a list's filter field on `/`, the way every list page in the app does.
 * Ignored while the user is already typing somewhere, so `/` still types a
 * slash into a text field.
 */
export function useListFilterHotkey(ref: RefObject<HTMLInputElement | null>) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target?.isContentEditable ?? false);
      if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        ref.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [ref]);
}

/** Filter/search rail: icon, field, clear-or-hint, and a live count. */
export function ListSearchRail({
  inputRef,
  value,
  onValueChange,
  placeholder,
  label,
  count,
}: {
  inputRef?: Ref<HTMLInputElement>;
  value: string;
  onValueChange: (value: string) => void;
  placeholder: string;
  /** Accessible name for the field; also names its clear button. */
  label: string;
  /** Right-aligned tally, e.g. "6 rules" or "12 / 340". */
  count: ReactNode;
}) {
  const { t } = useLingui();

  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-surface-border px-3 py-2.5 sm:px-4">
      <label className="input min-w-0 grow gap-2.5 pr-2">
        <Search size={16} strokeWidth={2} className="shrink-0 text-content-subtle" />
        <input
          ref={inputRef}
          type="search"
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          placeholder={placeholder}
          aria-label={label}
          className="min-w-0 grow bg-transparent text-sm text-content outline-none placeholder:text-content-subtle"
        />
        {value.trim() ? (
          <button
            type="button"
            onClick={() => onValueChange('')}
            className="btn btn-soft btn-circle btn-xs"
            aria-label={t`Clear ${label}`}
          >
            <X size={14} />
          </button>
        ) : (
          <kbd className="kbd kbd-sm hidden text-content-subtle sm:inline-flex">/</kbd>
        )}
      </label>
      <span className="hidden shrink-0 text-sm tabular-nums text-content-subtle sm:inline">
        {count}
      </span>
    </div>
  );
}

/** The card's only scroll region. Everything else in the card stays pinned. */
export function ListScroll({ children }: { children: ReactNode }) {
  return <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">{children}</div>;
}

/** Centered message for an empty or no-match list, filling the scroll region. */
export function ListEmpty({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 py-16 text-center">
      {children}
    </div>
  );
}

/** Footer pinned to the card's bottom edge — a tally, a note, or pagination. */
export function ListFooter({ children }: { children: ReactNode }) {
  return (
    <div className="mt-auto flex shrink-0 items-center gap-3 border-t border-surface-border px-3 py-2.5 sm:px-4">
      {children}
    </div>
  );
}

/**
 * Header cell of a list table. Sticky, so column labels survive scrolling.
 *
 * Applied per `<th>` rather than to `<thead>` or `<tr>`: WebKit — which is what
 * Tauri renders with on Linux — only honours `position: sticky` on the cells,
 * and a border on a non-sticky ancestor scrolls away from the labels it sits
 * under. Callers add their own horizontal padding per column.
 *
 * Filled with the region's own colour, not a panel's: the list no longer draws
 * a card, so a panel fill here would slide a lighter strip over the region as
 * rows scroll beneath it.
 */
export const LIST_HEAD_CELL =
  'sticky top-0 z-10 border-b border-surface-border bg-surface-shell py-2 text-left text-xs font-semibold uppercase tracking-wider text-content-subtle';

/** A body row: hairline rule below, hover tint, no jumping on hover. */
export const LIST_ROW =
  'border-b border-surface-border/60 transition-colors hover:bg-surface-hover';

/**
 * Column labels for a list that isn't a `<table>`. Same weight, tracking, and
 * rule as `LIST_HEAD_CELL`, so a `<ul>`-based list still reads as one family.
 */
export const LIST_HEAD_BAR =
  'flex items-center gap-4 border-b border-surface-border bg-surface-shell px-3 py-2 text-xs font-semibold uppercase tracking-wider text-content-subtle sm:px-4';

/**
 * Row action button. Hidden until the row is hovered or focused, so the eye
 * reads the row's content before its controls — but always visible at touch
 * widths, where there is no hover to reveal it. Add `btn-soft`, or
 * `btn-soft btn-error` for a destructive one.
 */
export const LIST_ROW_ACTION =
  'btn btn-sm btn-square opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100';
