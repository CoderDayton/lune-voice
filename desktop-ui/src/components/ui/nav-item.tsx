// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Shared look for navigation rows: the sidebar's page links and Settings'
 * section rail. Each keeps its own element (a router `Link`, a tab button) and
 * layout; the frame, the active state, the accent bar and the icon colour come
 * from here so the two read as one family.
 *
 * Active is a neutral pill with the icon's chip lit in primary. Primary stays
 * out of the label, so the only loud colour left in the sidebar is the mic
 * button.
 */
export const NAV_ITEM_BASE =
  'group relative flex items-center gap-3 rounded-full text-sm transition-colors focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary';

export function navItemState(active: boolean): string {
  return active
    ? 'bg-surface-nav-active font-medium text-content-strong ring-1 ring-surface-border'
    : 'text-content-muted hover:bg-surface-hover hover:text-content';
}

export function navIconClass(active: boolean): string {
  return `shrink-0 transition-colors ${
    active ? 'text-primary' : 'text-content-subtle group-hover:text-content-muted'
  }`;
}

/**
 * Round chip behind a row's icon. Lit with a soft primary wash when the row is
 * active, so the page marker lives on the icon instead of a bar on the edge.
 */
export function NavBead({
  active,
  className = '',
  children,
}: {
  active: boolean;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      aria-hidden="true"
      className={`flex size-8 shrink-0 items-center justify-center rounded-full transition-colors ${
        active
          ? 'bg-primary/15 ring-1 ring-primary/30'
          : 'bg-transparent group-hover:bg-surface-hover'
      } ${className}`}
    >
      {children}
    </span>
  );
}
