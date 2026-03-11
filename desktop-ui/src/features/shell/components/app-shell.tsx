import { Outlet, useRouterState } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { Topbar } from "./topbar";
import { findNavigationItem } from "../config/navigation";

export function AppShell() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const currentPage = findNavigationItem(pathname);

  return (
    <div className="app-shell-root">
      <div className="app-shell-backdrop" />

      <Sidebar />

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
