import { useLingui } from '@lingui/react/macro';
import { Outlet, useRouterState } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import { Menu } from 'lucide-react';
import { useEffect, useEffectEvent, useRef } from 'react';
import { ErrorToast } from '../../../components/feedback';
import type { SessionStreamEvent } from '../../../lib/backend/types';
import { workspaceActions, workspaceStore } from '../../../store/workspace-store';
import { UpdateDialog } from '../../updates/update-dialog';
import { useFirstRunGate } from '../../welcome/use-model-setup';
import { WelcomeLayout } from '../../welcome/welcome-layout';
import { WelcomePage } from '../../welcome/welcome-page';
import { DRAWER_ID } from '../config/drawer';
import { findNavigationItem } from '../config/navigation';
import { useAppShortcuts } from '../hooks/use-app-shortcuts';
import { startWorkspaceBridge } from '../workspace-bridge';
import { SidebarContent } from './sidebar';

export function AppShell() {
  const { t } = useLingui();
  useAppShortcuts();
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  });
  const page = findNavigationItem(pathname);
  const firstRun = useFirstRunGate();

  const onSessionEvent = useEffectEvent((event: SessionStreamEvent) => {
    workspaceActions.applySessionEvent(event);
  });

  const drawerRef = useRef<HTMLInputElement>(null);

  // Auto-close drawer overlay when crossing the lg breakpoint
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px)');
    const handle = (e: MediaQueryListEvent) => {
      if (drawerRef.current) drawerRef.current.checked = e.matches;
    };
    mq.addEventListener('change', handle);
    return () => mq.removeEventListener('change', handle);
  }, []);

  // Workspace bridge — only start once setup is complete so we don't race
  // against backend init while the user is still picking models.
  useEffect(() => {
    if (firstRun.status !== 'ready') return;
    return startWorkspaceBridge({
      onSessionEvent,
      onSnapshotError: (error) => {
        workspaceActions.setError(
          error instanceof Error ? error.message : 'Failed to reload workspace',
        );
      },
      onConnectionError: (scope, error) => {
        console.error(`[app-shell] ${scope} bridge failed`, error);
      },
    });
  }, [firstRun.status]);

  if (firstRun.status === 'checking') {
    return (
      <div className="welcome-shell relative h-dvh overflow-hidden">
        <div className="app-shell-backdrop" aria-hidden="true" />
        <div className="relative z-10 flex h-full items-center justify-center">
          <span className="loading loading-spinner loading-lg text-primary" />
        </div>
      </div>
    );
  }

  if (firstRun.status === 'needed') {
    return (
      <WelcomeLayout>
        <ErrorToast />
        <WelcomePage onFinish={firstRun.finish} />
      </WelcomeLayout>
    );
  }

  return (
    <div className="app-shell-root drawer lg:drawer-open fixed inset-0">
      <input
        ref={drawerRef}
        id={DRAWER_ID}
        type="checkbox"
        className="drawer-toggle"
        defaultChecked={typeof window !== 'undefined' && window.innerWidth >= 1024}
      />

      {/* Main content — must come before drawer-side per DaisyUI spec */}
      <div className="drawer-content relative z-10 flex flex-col min-h-0">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 sm:focus:top-4 sm:focus:left-4 focus:z-50 focus:btn focus:btn-primary focus:btn-sm"
        >
          {t`Skip to main content`}
        </a>
        <div className="app-shell-backdrop" />
        <ErrorToast />
        {/* Mounted once, at the shell, so the tray item, the Settings button
            and the sidebar quick action all report into the same modal — and
            so an update found while the window was closed is still waiting
            here when it opens. */}
        <UpdateDialog />

        {/* Mobile topbar */}
        <div className="flex items-center gap-2 px-3 py-2 lg:hidden border-b border-surface-border bg-surface-glass backdrop-blur-sm">
          <label htmlFor={DRAWER_ID} className="btn btn-soft btn-sm" aria-label={t`Open menu`}>
            <Menu size={20} strokeWidth={1.5} />
          </label>
          <span className="font-display tracking-heading text-base font-medium text-content-default select-none">
            {page.label}
          </span>
          <div className="ml-auto">
            <ShellStatusChip />
          </div>
        </div>

        <main id="main-content" className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className={page.to === '/' ? 'app-scroll app-scroll-fixed' : 'app-scroll'}>
            <div className={page.to === '/' ? 'app-frame app-frame-fill' : 'app-frame'}>
              {/* Page hero — Home leads with its bento board; every other
                  page gets its title + description back as an in-page header. */}
              {page.to !== '/' && (
                <header className="mb-7 select-none">
                  <h1 className="font-display text-display-lg font-semibold tracking-display text-content-strong">
                    {page.label}
                  </h1>
                  <p className="mt-1.5 text-sm text-content-subtle">{page.description}</p>
                </header>
              )}
              <Outlet />
            </div>
          </div>
        </main>
      </div>

      {/* Sidebar drawer */}
      <div className="drawer-side z-40 overflow-x-hidden">
        <label htmlFor={DRAWER_ID} aria-label={t`Close sidebar`} className="drawer-overlay" />
        <aside className="sidebar-surface flex is-drawer-close:w-[5rem] is-drawer-open:w-[18.5rem] min-h-full flex-col overflow-hidden transition-[width] duration-200">
          <SidebarContent />
        </aside>
      </div>
    </div>
  );
}

/** Live service-status chip for the desktop command bar. Subscribes to the
 *  runtime status only, so it re-renders independently of route changes. */
export function ShellStatusChip() {
  const { t } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);

  const isActive = serviceStatus === 'listening' || serviceStatus === 'processing';
  const isConnected = serviceStatus === 'ready' || isActive;
  const isConnecting = serviceStatus === 'Connecting';
  const dotClass = isConnecting ? 'bg-warning' : isConnected ? 'bg-success' : 'bg-error';
  const label = isConnecting
    ? t`Connecting`
    : serviceStatus === 'processing'
      ? t`Processing`
      : serviceStatus === 'listening'
        ? t`Listening`
        : isConnected
          ? t`Ready`
          : t`Offline`;

  return (
    <div className="flex shrink-0 items-center gap-2 rounded-selector border border-surface-border bg-surface-elevated px-3 py-1.5 text-sm text-content-muted select-none">
      <span className="relative inline-flex size-2 items-center justify-center">
        {isActive && <span className={`status-pulse absolute inset-0 rounded-full ${dotClass}`} />}
        <span className={`relative size-2 rounded-full ${dotClass}`} />
      </span>
      {label}
    </div>
  );
}
