import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { cn } from "../../../lib/cn";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { navigationItems } from "../config/navigation";
import { ChevronLeftIcon, ChevronRightIcon, LuneLogo } from "../config/icons";
import { useLingui } from "@lingui/react/macro";

export function Sidebar() {
  const { t } = useLingui();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const collapsed = useStore(workspaceStore, (state) => state.sidebarCollapsed);

  return (
    <aside
      className="sidebar-surface relative z-10 hidden md:flex shrink-0 flex-col gap-6 py-4 px-3 overflow-y-auto transition-[width] duration-200"
      style={{
        width: collapsed ? "var(--size-sidebar-collapsed)" : "var(--size-sidebar)",
      }}
    >
      <div className="flex items-center gap-3 px-2">
        <div className="brand-mark">
          <LuneLogo className="text-white" />
        </div>
        {!collapsed ? (
          <span className="brand-title">{t`Lune`}</span>
        ) : null}
      </div>

      <nav className="flex flex-col gap-1">
        {navigationItems.map(({ to, label, icon: Icon }) => {
          const active = to === "/" ? pathname === "/" : pathname.startsWith(to);

          return (
            <Link
              key={to}
              to={to}
              className={cn(active ? "nav-link-active" : "nav-link-idle")}
              title={collapsed ? label : undefined}
            >
              <Icon className={cn("nav-icon", active ? "text-secondary" : "")} />
              {!collapsed ? (
                <span className="truncate text-sm font-medium">{label}</span>
              ) : null}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto">
        <button
          type="button"
          className={cn("nav-link-idle w-full")}
          onClick={workspaceActions.toggleSidebar}
          aria-label={collapsed ? t`Expand sidebar` : t`Collapse sidebar`}
        >
          {collapsed
            ? <ChevronRightIcon className="nav-icon" />
            : <ChevronLeftIcon className="nav-icon" />}
          {!collapsed ? (
            <span className="truncate text-sm font-medium">{t`Collapse`}</span>
          ) : null}
        </button>
      </div>
    </aside>
  );
}
