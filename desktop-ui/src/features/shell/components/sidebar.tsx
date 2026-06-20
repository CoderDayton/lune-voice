import { useLingui } from '@lingui/react/macro';
import { Link, useRouterState } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import { invoke } from '@tauri-apps/api/core';
import {
  ChevronsUpDown,
  CircleUser,
  PanelLeftClose,
  PanelLeftOpen,
  Power,
  SunMoon,
} from 'lucide-react';
import luneLogo from '../../../assets/lune-voice-logo.png';
import { setThemePreference } from '../../../lib/theme';
import { workspaceStore } from '../../../store/workspace-store';
import { navigationGroups } from '../config/navigation';
import { DRAWER_ID } from './app-shell';
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
      {/* Brand header — org-switcher style block. When the sidebar is
          collapsed the logo doubles as the expand toggle: hovering it
          crossfades to the expand icon. */}
      <div className="flex w-full items-center gap-3 px-3 py-4 select-none is-drawer-close:justify-center is-drawer-close:px-2 sm:px-3.5 sm:py-5">
        <label
          htmlFor={DRAWER_ID}
          aria-label={t`Expand sidebar`}
          className="group/logo relative shrink-0 is-drawer-close:cursor-pointer is-drawer-open:pointer-events-none"
        >
          <div className="size-9 rounded-field ring-1 ring-surface-border-strong overflow-hidden sm:size-10">
            <img
              src={luneLogo}
              alt=""
              className="size-full object-cover pointer-events-none transition-opacity duration-150 is-drawer-close:group-hover/logo:opacity-0"
              draggable={false}
            />
          </div>
          <span
            aria-hidden="true"
            className="absolute inset-0 hidden items-center justify-center text-content opacity-0 transition-opacity duration-150 is-drawer-close:flex is-drawer-close:group-hover/logo:opacity-100"
          >
            <PanelLeftOpen size={20} strokeWidth={2} />
          </span>
          <span
            className={`absolute -bottom-0.5 -right-0.5 block size-2.5 rounded-full ring-2 ring-base-100 ${isActive ? 'bg-success status-pulse' : isConnecting ? 'bg-warning' : isConnected ? 'bg-success/60' : 'bg-base-content/25'}`}
            aria-hidden="true"
          />
        </label>
        <div className="min-w-0 flex-1 is-drawer-close:hidden">
          <span className="block text-lg leading-tight tracking-tight">
            <span className="font-bold">Lune</span>
            <span className="font-light text-content-muted"> Voice</span>
          </span>
          <span className="block text-[0.8rem] text-content-subtle">{t`Offline dictation`}</span>
        </div>
        <label
          htmlFor={DRAWER_ID}
          className="ml-auto cursor-pointer rounded-field p-1.5 text-content-subtle transition-colors hover:bg-surface-hover hover:text-content is-drawer-close:hidden"
          aria-label={t`Collapse sidebar`}
        >
          <PanelLeftClose size={18} strokeWidth={2} />
        </label>
      </div>

      <div className="mx-3 border-t border-surface-border sm:mx-3.5" />

      {/* Grouped navigation */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden px-2 py-3 is-drawer-close:px-0 sm:px-3">
        {navigationGroups.map((group) => (
          <div key={group.label} className="mb-5 last:mb-0">
            <p className="section-label mb-2 px-3 is-drawer-close:hidden">{group.label}</p>
            <ul className="flex flex-col gap-0.5">
              {group.items.map((item) => (
                <NavRow key={item.to} item={item} pathname={pathname} />
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Primary action — full button when expanded, compact mic tile when collapsed */}
      <SidebarQuickStart />

      {/* Account / status card */}
      <div className="mx-3 mb-2 sm:mx-3.5 sm:mb-3">
        <div className="mb-3 border-t border-surface-border" />
        <SidebarUserCard
          isActive={isActive}
          isConnected={isConnected}
          isConnecting={isConnecting}
        />
      </div>
    </>
  );
}

function NavRow({
  item,
  pathname,
}: {
  item: (typeof navigationGroups)[number]['items'][number];
  pathname: string;
}) {
  const { to, label, icon: Icon } = item;
  const active = to === '/' ? pathname === '/' : pathname.startsWith(to);
  return (
    <li>
      <Link
        to={to}
        aria-current={active ? 'page' : undefined}
        title={label}
        className={`group relative flex w-full items-center gap-3 rounded-field px-3 py-2.5 text-[0.95rem] transition-colors is-drawer-close:mx-auto is-drawer-close:size-10 is-drawer-close:justify-center is-drawer-close:gap-0 is-drawer-close:px-0 is-drawer-close:py-0 ${
          active
            ? 'bg-surface-nav-active text-content-strong font-medium'
            : 'text-content-muted hover:bg-surface-hover hover:text-content'
        }`}
      >
        {active ? (
          <span
            aria-hidden="true"
            className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-primary is-drawer-close:hidden"
          />
        ) : null}
        <Icon
          size={19}
          strokeWidth={active ? 2.2 : 1.7}
          className={active ? 'text-primary' : 'text-content-subtle group-hover:text-content-muted'}
        />
        <span className="truncate is-drawer-close:hidden">{label}</span>
      </Link>
    </li>
  );
}

function SidebarUserCard({
  isActive,
  isConnected,
  isConnecting,
}: {
  isActive: boolean;
  isConnected: boolean;
  isConnecting: boolean;
}) {
  const { t } = useLingui();
  const dotClass = isActive
    ? 'bg-success status-pulse'
    : isConnecting
      ? 'bg-warning'
      : isConnected
        ? 'bg-success/60'
        : 'bg-base-content/25';
  const statusLabel = isActive
    ? t`Listening`
    : isConnecting
      ? t`Connecting`
      : isConnected
        ? t`Ready`
        : t`Offline`;

  // DaisyUI dropdowns are focus-driven; blur the trigger so the menu closes
  // after an action instead of lingering over the sidebar.
  const closeDropdown = () => {
    (document.activeElement as HTMLElement | null)?.blur();
  };

  const toggleTheme = () => {
    const scheme = document.documentElement.getAttribute('data-color-scheme');
    setThemePreference(scheme === 'light' ? 'lune' : 'lune-light');
  };

  const quitApp = () => {
    void invoke('quit_app').catch(() => {});
  };

  return (
    <div className="dropdown dropdown-top w-full is-drawer-close:dropdown-right is-drawer-close:dropdown-end">
      <button
        type="button"
        className="flex w-full items-center gap-3 rounded-field border border-surface-border bg-surface-elevated px-2.5 py-2.5 text-left transition-colors hover:bg-surface-card-hover hover:border-surface-border-strong focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary is-drawer-close:mx-auto is-drawer-close:size-10 is-drawer-close:justify-center is-drawer-close:gap-0 is-drawer-close:p-0 sm:px-3 is-drawer-close:sm:p-0"
        aria-label={t`Account and status`}
        onMouseDown={(e) => {
          // DaisyUI dropdowns open on focus, so a second click on the
          // already-open trigger wouldn't close it (it stays focused).
          // If it's focused, blur and swallow the click so it can't
          // immediately re-focus and reopen.
          if (document.activeElement === e.currentTarget) {
            e.preventDefault();
            e.currentTarget.blur();
          }
        }}
      >
        <span className="relative shrink-0">
          <span className="flex size-8 items-center justify-center rounded-field bg-surface-wash text-content-muted ring-1 ring-surface-border-strong">
            <CircleUser size={20} strokeWidth={1.6} />
          </span>
          <span
            className={`absolute -bottom-0.5 -right-0.5 size-2.5 rounded-full ring-2 ring-surface-elevated ${dotClass}`}
            aria-hidden="true"
          />
        </span>
        <span className="min-w-0 flex-1 is-drawer-close:hidden">
          <span className="block truncate text-sm font-medium text-content">{t`Lune Voice`}</span>
          <span className="block truncate text-xs text-content-subtle">
            {t`Local`} · {statusLabel}
          </span>
        </span>
        <ChevronsUpDown
          size={15}
          strokeWidth={1.8}
          className="shrink-0 text-content-subtle is-drawer-close:hidden"
        />
      </button>
      <ul className="dropdown-content menu z-popover mb-2 w-full gap-0.5 rounded-box border border-surface-border bg-surface-elevated p-2 shadow-overlay is-drawer-close:mb-0 is-drawer-close:ml-2 is-drawer-close:w-56 is-drawer-close:top-auto is-drawer-close:bottom-0">
        <li>
          <button
            type="button"
            onClick={() => {
              closeDropdown();
              toggleTheme();
            }}
          >
            <SunMoon size={16} strokeWidth={1.8} />
            {t`Toggle theme`}
          </button>
        </li>
        <div aria-hidden="true" className="mx-2 my-1 border-t border-surface-border" />
        <li>
          <button
            type="button"
            className="text-error hover:bg-error/10"
            onClick={() => {
              closeDropdown();
              quitApp();
            }}
          >
            <Power size={16} strokeWidth={1.8} />
            {t`Quit Lune Voice`}
          </button>
        </li>
      </ul>
    </div>
  );
}
