import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { navigationItems } from "../config/navigation";
import { ChevronLeftIcon, ChevronRightIcon } from "../config/icons";
import luneLogo from "../../../assets/lune-voice-logo.png";
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
        <div className="size-10 shrink-0 select-none pointer-events-none" aria-hidden="true">
          <div className="size-full rounded-xl overflow-hidden">
            <img src={luneLogo} alt="" className="size-full object-cover" draggable={false} />
          </div>
        </div>
        {!collapsed ? (
          <span className="brand-title">{t`Lune Voice`}</span>
        ) : null}
      </div>

      <ul className="menu menu-sm gap-1 px-0">
        {navigationItems.map(({ to, label, icon: Icon }) => {
          const active = to === "/" ? pathname === "/" : pathname.startsWith(to);
          return (
            <li key={to}>
              <Link
                to={to}
                className={`${active ? "menu-active" : ""}${collapsed ? " tooltip tooltip-right" : ""}`}
                data-tip={collapsed ? label : undefined}
              >
                <Icon className={`nav-icon${active ? " text-secondary" : ""}`} />
                {!collapsed ? <span className="truncate">{label}</span> : null}
              </Link>
            </li>
          );
        })}
      </ul>

      <ul className="menu menu-sm px-0 mt-auto">
        <li>
          <button
            type="button"
            onClick={workspaceActions.toggleSidebar}
            aria-label={collapsed ? t`Expand sidebar` : t`Collapse sidebar`}
          >
            {collapsed ? <ChevronRightIcon className="nav-icon" /> : <ChevronLeftIcon className="nav-icon" />}
            {!collapsed ? <span className="truncate">{t`Collapse`}</span> : null}
          </button>
        </li>
      </ul>
    </aside>
  );
}
