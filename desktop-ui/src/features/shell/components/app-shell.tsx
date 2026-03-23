import { useEffect, useEffectEvent } from "react";
import { Outlet } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { MobileNav } from "./mobile-nav";
import { Topbar } from "./topbar";
import { ErrorToast } from "../../../components/feedback";
import { useAppShortcuts } from "../hooks/use-app-shortcuts";
import type { SessionStreamEvent } from "../../../lib/backend/types";
import { workspaceActions } from "../../../store/workspace-store";
import { startWorkspaceBridge } from "../workspace-bridge";

export function AppShell() {
  useAppShortcuts();

  const onSessionEvent = useEffectEvent((event: SessionStreamEvent) => {
    workspaceActions.applySessionEvent(event);
  });

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
    <div className="app-shell-root">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 sm:focus:top-4 sm:focus:left-4 focus:z-50 focus:btn focus:btn-primary focus:btn-sm">
        Skip to main content
      </a>
      <div className="app-shell-backdrop" />

      <Sidebar />
      <MobileNav />
      <ErrorToast />

      <main id="main-content" className="app-main">
        <Topbar />

        <div className="app-scroll">
          <div className="app-frame">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
