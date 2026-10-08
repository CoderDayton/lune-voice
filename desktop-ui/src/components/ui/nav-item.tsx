// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Shared look for navigation rows: the sidebar's page links and Settings'
 * section rail. Each keeps its own element (a router `Link`, a tab button) and
 * layout; the frame, the active state, the accent bar and the icon colour come
 * from here so the two read as one family.
 *
 * Active is a neutral fill with a primary bar and icon. Primary stays out of
 * the label, so the only loud colour left in the sidebar is the mic button.
 */
export const NAV_ITEM_BASE =
  'group relative flex items-center gap-3 rounded-field text-sm transition-colors focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary';

export function navItemState(active: boolean): string {
  return active
    ? 'bg-surface-nav-active font-medium text-content-strong'
    : 'text-content-muted hover:bg-surface-hover hover:text-content';
}

export function navIconClass(active: boolean): string {
  return `shrink-0 transition-colors ${
    active ? 'text-primary' : 'text-content-subtle group-hover:text-content-muted'
  }`;
}

/** Rounded accent bar on the row's leading edge. `className` can add a breakpoint. */
export function NavActiveBar({ className = '' }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={`absolute inset-y-2 left-0 w-[3px] rounded-full bg-primary ${className}`}
    />
  );
}
