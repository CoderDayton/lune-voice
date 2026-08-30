import { useLingui } from '@lingui/react/macro';
import { formatForDisplay } from '@tanstack/react-hotkeys';
import { Link, useRouterState } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import luneLogo from '../../../assets/lune-voice-logo.png';
import { workspaceStore } from '../../../store/workspace-store';
import { DRAWER_ID } from '../config/drawer';
import { type NavigationItem, navigationGroups } from '../config/navigation';
import { SidebarQuickActions } from './sidebar-quick-actions';
import { SidebarQuickStart } from './sidebar-widgets';

export function SidebarContent() {
  const { t } = useLingui();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);

  const isActive = serviceStatus === 'listening' || serviceStatus === 'processing';
  const isConnected = serviceStatus === 'ready' || isActive;
  const isConnecting = serviceStatus === 'Connecting';

  return (
    <>
      {/* Brand header. When the sidebar is collapsed the logo doubles as the
          expand toggle: hovering it crossfades to the expand icon. */}
      <div className="flex w-full items-center gap-3 border-b border-surface-border py-4 pr-4 pl-4 select-none is-drawer-close:justify-center is-drawer-close:px-0">
        <label
          htmlFor={DRAWER_ID}
          aria-label={t`Expand sidebar`}
          className="group/logo relative shrink-0 is-drawer-close:cursor-pointer is-drawer-open:pointer-events-none"
        >
          <div className="size-10 overflow-hidden rounded-field ring-1 ring-surface-border-strong">
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
            <PanelLeftOpen size={22} strokeWidth={2} />
          </span>
          <span
            className={`absolute -right-px -bottom-px block size-2.5 rounded-full ring-2 ring-surface-ground ${isActive ? 'bg-success status-pulse' : isConnecting ? 'bg-warning' : isConnected ? 'bg-success/70' : 'bg-content-dim'}`}
            aria-hidden="true"
          />
        </label>
        <span className="min-w-0 flex-1 truncate text-base leading-tight tracking-heading text-content-strong is-drawer-close:hidden">
          <span className="font-semibold">Lune</span>
          <span className="font-light text-content-muted"> Voice</span>
        </span>
        <label
          htmlFor={DRAWER_ID}
          className="cursor-pointer rounded-field p-2 text-content-subtle transition-colors hover:bg-surface-hover hover:text-content is-drawer-close:hidden"
          aria-label={t`Collapse sidebar`}
        >
          <PanelLeftClose size={20} strokeWidth={1.8} />
        </label>
      </div>

      {/* Navigation — groups are separated by a hairline, not a label. */}
      <nav className="flex flex-1 flex-col overflow-x-hidden overflow-y-auto py-2">
        {navigationGroups.map((group, index) => (
          <div key={group.label}>
            {index > 0 ? (
              <div
                aria-hidden="true"
                className="my-2.5 border-t border-surface-border mx-5 is-drawer-close:mx-auto is-drawer-close:w-7"
              />
            ) : null}
            <ul aria-label={group.label} className="flex flex-col">
              {group.items.map((item) => (
                <NavRow key={item.to} item={item} pathname={pathname} />
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Primary action — flush footer band, mic tile only when collapsed */}
      <SidebarQuickStart />

      {/* Secondary: theme, shortcuts, updates, help, quit */}
      <SidebarQuickActions />
    </>
  );
}

function NavRow({ item, pathname }: { item: NavigationItem; pathname: string }) {
  const { to, label, shortcut, icon: Icon } = item;
  const active = to === '/' ? pathname === '/' : pathname.startsWith(to);
  const hint = formatForDisplay(shortcut);

  return (
    <li>
      <Link
        to={to}
        aria-current={active ? 'page' : undefined}
        title={`${label} · ${hint}`}
        className={`group relative flex h-12 w-full items-center gap-3 px-5 text-sm transition-colors is-drawer-close:h-12 is-drawer-close:justify-center is-drawer-close:gap-0 is-drawer-close:px-0 ${
          active
            ? 'bg-primary/10 font-medium text-content-strong'
            : 'text-content-muted hover:bg-surface-hover hover:text-content'
        }`}
      >
        <span
          aria-hidden="true"
          className={`absolute inset-y-0 left-0 w-0.5 transition-colors ${
            active ? 'bg-primary' : 'bg-transparent group-hover:bg-surface-border-strong'
          }`}
        />
        <Icon
          size={18}
          strokeWidth={active ? 2 : 1.7}
          className={`shrink-0 is-drawer-close:size-[1.1875rem] ${
            active ? 'text-primary' : 'text-content-subtle group-hover:text-content-muted'
          }`}
        />
        <span className="truncate is-drawer-close:hidden">{label}</span>
        <span
          aria-hidden="true"
          className={`ml-auto font-mono text-xs is-drawer-close:hidden ${
            active ? 'text-primary' : 'text-content-subtle group-hover:text-content-muted'
          }`}
        >
          {hint}
        </span>
      </Link>
    </li>
  );
}
