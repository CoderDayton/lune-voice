import { useEffect, useEffectEvent } from "react";
import { Outlet } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { MobileNav } from "./mobile-nav";
import { Topbar } from "./topbar";
import { ErrorToast } from "../../../components/feedback";
import { useAppShortcuts } from "../hooks/use-app-shortcuts";
import { attachSessionStream, listenToWorkspaceInvalidation } from "../../../lib/backend/stream";
import type { SessionStreamEvent } from "../../../lib/backend/types";
import {
  loadWorkspaceSnapshot,
  workspaceActions,
  workspaceStore,
} from "../../../store/workspace-store";

export function AppShell() {
  useAppShortcuts();

  useEffect(() => {
    // Skip if already hydrated by bootstrap in main.tsx
    if (workspaceStore.state.runtime.loaded) return;
    let cancelled = false;
    loadWorkspaceSnapshot().catch((e) => {
      if (cancelled) return;
      workspaceActions.setError(e instanceof Error ? e.message : String(e));
    });
    return () => { cancelled = true; };
  }, []);

  const onSessionEvent = useEffectEvent((event: SessionStreamEvent) => {
    workspaceActions.applySessionEvent(event);
  });

  useEffect(() => {
    const disposers: Array<() => void> = [];
    let cancelled = false;

    async function connectSessionStream() {
      try {
        const detach = await attachSessionStream(onSessionEvent);
        if (cancelled) { detach(); return; }
        disposers.push(detach);
      } catch (e) {
        console.error("[app-shell] session stream failed", e);
      }
    }

    async function connectHistoryInvalidation() {
      try {
        const detach = await listenToWorkspaceInvalidation("lune://history-changed", () => {
          loadWorkspaceSnapshot().catch((e) => {
            workspaceActions.setError(e instanceof Error ? e.message : "Failed to reload workspace");
          });
        });
        if (cancelled) { detach(); return; }
        disposers.push(detach);
      } catch (e) {
        console.error("[app-shell] history invalidation failed", e);
      }
    }

    connectSessionStream();
    connectHistoryInvalidation();

    return () => {
      cancelled = true;
      for (const dispose of disposers) dispose();
    };
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
