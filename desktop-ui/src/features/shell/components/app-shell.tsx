import { useEffect, useEffectEvent, useRef } from "react";
import { Outlet, useRouterState } from "@tanstack/react-router";
import { Menu } from "lucide-react";
import { SidebarContent } from "./sidebar";
import { ErrorToast } from "../../../components/feedback";
import { useAppShortcuts } from "../hooks/use-app-shortcuts";
import { findNavigationItem } from "../config/navigation";
import type { SessionStreamEvent } from "../../../lib/backend/types";
import { workspaceActions } from "../../../store/workspace-store";
import { startWorkspaceBridge } from "../workspace-bridge";

export const DRAWER_ID = "app-drawer";

export function AppShell() {
  useAppShortcuts();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const page = findNavigationItem(pathname);

  const onSessionEvent = useEffectEvent((event: SessionStreamEvent) => {
    workspaceActions.applySessionEvent(event);
  });

  const drawerRef = useRef<HTMLInputElement>(null);

  // Auto-close drawer overlay when crossing the lg breakpoint
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const handle = (e: MediaQueryListEvent) => {
      if (drawerRef.current) drawerRef.current.checked = e.matches;
    };
    mq.addEventListener("change", handle);
    return () => mq.removeEventListener("change", handle);
  }, []);

  useEffect(() => {
    return startWorkspaceBridge({
      onSessionEvent,
      onSnapshotError: (error) => {
        workspaceActions.setError(
          error instanceof Error ? error.message : "Failed to reload workspace",
        );
      },
      onConnectionError: (scope, error) => {
        console.error(`[app-shell] ${scope} bridge failed`, error);
      },
    });
  }, []);

  return (
    <div className="app-shell-root drawer lg:drawer-open fixed inset-0">
      <input ref={drawerRef} id={DRAWER_ID} type="checkbox" className="drawer-toggle" defaultChecked={typeof window !== "undefined" && window.innerWidth >= 1024} />

      {/* Main content — must come before drawer-side per DaisyUI spec */}
      <div className="drawer-content flex flex-col min-h-0">
        <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 sm:focus:top-4 sm:focus:left-4 focus:z-50 focus:btn focus:btn-primary focus:btn-sm">
          Skip to main content
        </a>
        <div className="app-shell-backdrop" />
        <ErrorToast />

        {/* Mobile topbar */}
        <div className="flex items-center px-4 py-2 lg:hidden">
          <label htmlFor={DRAWER_ID} className="btn btn-ghost btn-sm" aria-label="Open menu">
            <Menu size={20} strokeWidth={1.5} />
          </label>
        </div>

        <main id="main-content" className="flex-1 overflow-y-auto">
          <div className="app-scroll">
            <div className="app-frame">
              <h1 className="mb-6 text-3xl font-bold tracking-tight text-balance select-none sm:text-3xl">
                {page.label}
              </h1>
              <Outlet />
            </div>
          </div>
        </main>
      </div>

      {/* Sidebar drawer */}
      <div className="drawer-side z-40 is-drawer-close:overflow-visible">
        <label htmlFor={DRAWER_ID} aria-label="Close sidebar" className="drawer-overlay" />
        <aside className="sidebar-surface flex is-drawer-close:w-[5rem] is-drawer-open:w-[18.5rem] min-h-full flex-col overflow-hidden transition-[width] duration-200">
          <SidebarContent />
        </aside>
      </div>
    </div>
  );
}
