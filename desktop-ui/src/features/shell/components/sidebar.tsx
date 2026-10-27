// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { formatForDisplay } from '@tanstack/react-hotkeys';
import { Link, useRouterState } from '@tanstack/react-router';
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import luneLogo from '../../../assets/lune-voice-logo.png';
import {
  NAV_ITEM_BASE,
  NavBead,
  navIconClass,
  navItemState,
} from '../../../components/ui/nav-item';
import { DRAWER_ID, SIDEBAR_NAV_ID } from '../config/drawer';
import { type NavigationItem, navigationGroups } from '../config/navigation';
import { useServiceStatus } from '../hooks/use-service-status';
import { RailTooltip } from './rail-tooltip';
import { SidebarQuickActions } from './sidebar-quick-actions';
import { SidebarQuickStart } from './sidebar-widgets';

export function SidebarContent() {
  const { t, i18n } = useLingui();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const { isActive, dotClass } = useServiceStatus();

  return (
    <>
      {/* Brand header. When the sidebar is collapsed the logo doubles as the
          expand toggle: hovering it crossfades to the expand icon. */}
      <div className="flex w-full items-center gap-3 px-4 pt-5 pb-4 select-none is-drawer-close:justify-center is-drawer-close:px-0">
        <label
          htmlFor={DRAWER_ID}
          aria-label={t`Expand sidebar`}
          className="group/logo relative shrink-0 is-drawer-close:cursor-pointer is-drawer-open:pointer-events-none"
        >
          <div className="size-9 overflow-hidden rounded-field ring-1 ring-surface-border-strong">
            <img
              src={luneLogo}
              alt=""
              className="pointer-events-none size-full object-cover transition-opacity duration-fast is-drawer-close:group-hover/logo:opacity-0"
              draggable={false}
            />
          </div>
          <span
            aria-hidden="true"
            className="absolute inset-0 hidden items-center justify-center text-content opacity-0 transition-opacity duration-fast is-drawer-close:flex is-drawer-close:group-hover/logo:opacity-100"
          >
            <PanelLeftOpen size={20} strokeWidth={2} className="rtl:-scale-x-100" />
          </span>
          <span
            className={`absolute -end-px -bottom-px block size-2.5 rounded-full ring-2 ring-surface-ground ${dotClass} ${isActive ? 'status-pulse' : ''}`}
            aria-hidden="true"
          />
        </label>
        <span className="min-w-0 flex-1 truncate font-display text-base leading-tight tracking-heading text-content-strong is-drawer-close:hidden">
          <span className="font-semibold">Lune</span>
          <span className="font-light text-content-muted"> Voice</span>
        </span>
        <label
          htmlFor={DRAWER_ID}
          className="cursor-pointer rounded-full p-2 text-content-subtle transition-colors hover:bg-surface-hover hover:text-content is-drawer-close:hidden"
          aria-label={t`Collapse sidebar`}
        >
          <PanelLeftClose size={20} strokeWidth={1.8} className="rtl:-scale-x-100" />
        </label>
      </div>

      {/* Navigation. Groups are separated by a hairline, not a label; the last
          group (Settings) sits at the foot of the list. The nav is focusable
          by script only: the mic button parks keyboard focus here. */}
      <nav
        id={SIDEBAR_NAV_ID}
        tabIndex={-1}
        className="flex flex-1 flex-col overflow-x-hidden overflow-y-auto py-1"
      >
        {navigationGroups.map((group, index) => (
          <div key={group.label} className={index > 0 ? 'mt-auto' : undefined}>
            {index > 0 ? (
              <div
                aria-hidden="true"
                className="mx-6 my-2 border-t border-surface-border is-drawer-close:mx-auto is-drawer-close:w-7"
              />
            ) : null}
            <ul aria-label={i18n._(group.name)} className="flex flex-col gap-0.5">
              {group.items.map((item) => (
                <NavRow key={item.to} item={item} pathname={pathname} />
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <div className="flex flex-col gap-1 px-3 pt-2 pb-3">
        {/* Primary action — the sidebar's one accent */}
        <SidebarQuickStart />

        {/* Secondary: theme, shortcuts, updates, help, quit */}
        <SidebarQuickActions />
      </div>
    </>
  );
}

function NavRow({ item, pathname }: { item: NavigationItem; pathname: string }) {
  const { i18n } = useLingui();
  const { to, shortcut, icon: Icon } = item;
  const label = i18n._(item.label);
  const active = to === '/' ? pathname === '/' : pathname.startsWith(to);
  const hint = formatForDisplay(shortcut);

  return (
    <li>
      <RailTooltip label={label} hint={hint}>
        <Link
          to={to}
          aria-current={active ? 'page' : undefined}
          // The label is hidden when collapsed; this keeps the link named.
          aria-label={label}
          className={`${NAV_ITEM_BASE} mx-3 h-12 pe-4 ps-2 ${navItemState(active)} is-drawer-close:justify-center is-drawer-close:gap-0 is-drawer-close:px-0`}
        >
          <NavBead active={active}>
            <Icon size={17} strokeWidth={active ? 2 : 1.7} className={navIconClass(active)} />
          </NavBead>
          <span className="flex min-w-0 flex-col is-drawer-close:hidden">
            <span className="truncate">{label}</span>
            {/* A bare "Settings" hides the models and shortcuts inside, so a
                page can carry a line saying what it holds. */}
            {item.blurb && (
              <span className="truncate text-xs font-normal text-content-subtle">
                {i18n._(item.blurb)}
              </span>
            )}
          </span>
          {/* Shortcut hint: out of the way until the row is hovered or focused. */}
          <span
            aria-hidden="true"
            className="ms-auto font-mono text-xs text-content-subtle opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 is-drawer-close:hidden"
          >
            {hint}
          </span>
        </Link>
      </RailTooltip>
    </li>
  );
}
