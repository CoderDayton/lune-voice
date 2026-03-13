import { useEffect } from "react";
import { Outlet, useRouterState } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { MobileNav } from "./mobile-nav";
import { Topbar } from "./topbar";
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
    void loadWorkspaceSnapshot();
  }, []);

  useEffect(() => {
    let disposeSession = () => {};
    let disposeHistory = () => {};

    void (async () => {
      disposeSession = await attachSessionStream((event) => {
        workspaceActions.applySessionEvent(event);
      });
      disposeHistory = await listenToWorkspaceInvalidation("lune://history-changed", () => {
        void loadWorkspaceSnapshot();
      });
    })();

    return () => {
      disposeSession();
      disposeHistory();
    };
  }, []);

  return (
    <div className="app-shell-root">
      <div className="app-shell-backdrop" />

      <Sidebar />
      <MobileNav />

      <main className="app-main">
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
