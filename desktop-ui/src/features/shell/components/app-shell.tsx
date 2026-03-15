import { useEffect } from "react";
import { Outlet, useRouterState } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { MobileNav } from "./mobile-nav";
import { Topbar } from "./topbar";
import { ErrorToast } from "../../../components/ui/error-toast";
import { findNavigationItem } from "../config/navigation";
import { attachSessionStream, listenToWorkspaceInvalidation } from "../../../lib/backend/stream";
import {
  loadWorkspaceSnapshot,
  workspaceActions,
} from "../../../store/workspace-store";

export function AppShell() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const currentPage = findNavigationItem(pathname);

  useEffect(() => {
    loadWorkspaceSnapshot().catch((e) => {
      workspaceActions.setError(e instanceof Error ? e.message : "Failed to load workspace");
    });
  }, []);

  useEffect(() => {
    const disposers: Array<() => void> = [];
    let cancelled = false;

    (async () => {
      try {
        const detachSession = await attachSessionStream((event) => {
          workspaceActions.applySessionEvent(event);
        });
        if (cancelled) { detachSession(); return; }
        disposers.push(detachSession);

        const detachHistory = await listenToWorkspaceInvalidation("lune://history-changed", () => {
          loadWorkspaceSnapshot().catch((e) => {
            workspaceActions.setError(e instanceof Error ? e.message : "Failed to reload workspace");
          });
        });
        if (cancelled) { detachHistory(); return; }
        disposers.push(detachHistory);
      } catch (e) {
        console.error("[app-shell] failed to attach IPC streams", e);
      }
    })();

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
        <Topbar page={currentPage} />

        <div className="app-scroll">
          <div className="app-frame">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
