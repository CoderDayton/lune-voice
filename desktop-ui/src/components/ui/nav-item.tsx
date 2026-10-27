// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Shared look for navigation rows: the sidebar's page links and Settings'
 * section rail. Each keeps its own element (a router `Link`, a tab button) and
 * layout; the frame, the active state, the accent bar and the icon colour come
 * from here so the two read as one family.
 *
 * Active is a pill washed in primary with the icon's chip lit to match; the
 * label stays in the strong text colour so it reads at a glance.
 */
export const NAV_ITEM_BASE =
  'group relative flex items-center gap-3 rounded-full text-base transition-colors focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary';

export function navItemState(active: boolean): string {
  return active
    ? 'bg-primary/10 font-medium text-content-strong ring-1 ring-primary/25'
    : 'text-content hover:bg-surface-hover hover:text-content-strong';
}

export function navIconClass(active: boolean): string {
  return `shrink-0 transition-colors ${
    active ? 'text-primary' : 'text-content-muted group-hover:text-content'
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
          ? 'bg-primary/20 ring-1 ring-primary/35'
          : 'bg-transparent group-hover:bg-surface-hover'
      } ${className}`}
    >
      {children}
    </span>
  );
}
