import { Link, useRouterState } from '@tanstack/react-router';
import { ChevronRight } from 'lucide-react';
import { findNavigationItem, navigationGroups } from '../config/navigation';
import { ShellStatusChip } from './app-shell';

/**
 * Desktop top bar — breadcrumb trail + page context + live status chip.
 *
 * Not currently mounted by the shell. Kept as a standalone component so the
 * markup survives intact and can be reintroduced (or redesigned) later without
 * digging it back out of git history.
 */
export function DesktopTopBar() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const page = findNavigationItem(pathname);
  // Breadcrumb trail: the group crumb links to that group's primary route.
  const breadcrumbGroup = navigationGroups.find((g) => g.label === page.group);
  const groupHome = breadcrumbGroup?.items[0]?.to ?? '/';

  return (
    <div className="hidden h-14 shrink-0 items-center gap-3 border-b border-surface-border bg-surface-glass px-4 lg:flex xl:px-6">
      <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1 text-sm select-none">
        <Link
          to={groupHome}
          className="shrink-0 rounded-field px-1.5 py-0.5 text-content-subtle transition-colors hover:bg-surface-hover hover:text-content"
        >
          {page.group}
        </Link>
        <ChevronRight size={14} strokeWidth={2} className="shrink-0 text-content-dim" />
        <Link
          to={page.to}
          aria-current="page"
          className="truncate rounded-field px-1.5 py-0.5 font-medium text-content-strong transition-colors hover:bg-surface-hover"
        >
          {page.label}
        </Link>
      </nav>
      <div className="hidden h-5 w-px shrink-0 bg-surface-border xl:block" />
      <p className="hidden min-w-0 truncate text-sm text-content-subtle select-none xl:block">
        {page.description}
      </p>
      <div className="ml-auto flex shrink-0 items-center gap-3">
        <span className="hidden text-sm text-content-dim select-none xl:block">
          {new Date().toLocaleDateString(undefined, {
            weekday: 'long',
            month: 'long',
            day: 'numeric',
          })}
        </span>
        <ShellStatusChip />
      </div>
    </div>
  );
}
