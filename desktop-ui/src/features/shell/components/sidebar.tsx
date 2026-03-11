import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { Button } from "../../../components/ui/button";
import { Pill } from "../../../components/ui/pill";
import { cn } from "../../../lib/cn";
import { authActions, authStore } from "../../../store/auth-store";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { findNavigationItem, navigationItems } from "../config/navigation";
import { ChevronLeftIcon, ChevronRightIcon, LuneLogo } from "../config/icons";

const statusItems = [
  { label: "Microphone", value: "Ready", tone: "text-brand-300" },
  { label: "Voice typing", value: "Fast + accurate", tone: "text-brand-400" },
  { label: "Privacy", value: "On this device", tone: "text-brand-500" },
];

export function Sidebar() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const collapsed = useStore(workspaceStore, (state) => state.sidebarCollapsed);
  const profile = useStore(authStore, (state) => state.profile);
  const current = findNavigationItem(pathname);
  const navigationGroups = [
    {
      label: "Use Lune",
      items: navigationItems.filter((item) => item.section === "Use Lune"),
    },
    {
      label: "Set up",
      items: navigationItems.filter((item) => item.section === "Set up"),
    },
  ];

  return (
    <aside
      className="surface-sidebar relative z-10 flex shrink-0 flex-col transition-[width] duration-200"
      style={{
        width: collapsed ? "var(--size-sidebar-collapsed)" : "var(--size-sidebar)",
        gap: "var(--space-sidebar-y)",
        padding: "var(--space-sidebar-y) var(--space-sidebar-x)",
      }}
    >
      <div className="flex items-center gap-3 px-2">
        <div className="brand-mark">
          <LuneLogo className="text-white" />
        </div>
        {!collapsed ? (
          <div className="min-w-0">
            <p className="app-kicker">Desktop Dictation</p>
            <h1 className="brand-title">Lune Voice</h1>
          </div>
        ) : null}
      </div>

      <div className="sidebar-divider" />

      <nav className="flex flex-col gap-4">
        {navigationGroups.map((group) => (
          <div key={group.label} className="space-y-1">
            {!collapsed ? <p className="app-kicker px-2">{group.label}</p> : null}
            {group.items.map(({ to, label, caption, icon: Icon }) => {
              const active = to === "/" ? pathname === "/" : pathname.startsWith(to);

              return (
                <Link
                  key={to}
                  to={to}
                  className={cn(active ? "nav-link-active" : "nav-link-idle")}
                  title={collapsed ? label : undefined}
                >
                  <Icon className={cn("nav-icon", active ? "text-brand-400" : "text-copy-400")} />
                  {!collapsed ? (
                    <div className="min-w-0">
                      <div className="truncate supporting-label text-copy-50">{label}</div>
                      <div className="truncate text-[var(--text-pill)] text-copy-500">{caption}</div>
                    </div>
                  ) : null}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {!collapsed ? (
        <div className="sidebar-card">
          <div className="sidebar-card-header">
            <div>
              <p className="app-kicker">Session posture</p>
              <h2 className="sidebar-card-title mt-2">{current.label} is ready</h2>
              <p className="sidebar-card-copy mt-1">
                Your recordings stay private on this computer, with quick cleanup before text is inserted.
              </p>
            </div>
            <Pill>Stable</Pill>
          </div>

          <div className="sidebar-list">
            {statusItems.map((item) => (
              <div key={item.label} className="sidebar-list-row">
                <span className="sidebar-key">{item.label}</span>
                <span className={cn("font-semibold", item.tone)}>{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="mt-auto space-y-3">
        {!collapsed ? (
          <div className="sidebar-card">
            <p className="app-kicker">Signed in</p>
            <h2 className="sidebar-card-title mt-2">{profile?.name ?? "Lune user"}</h2>
            <p className="sidebar-card-copy mt-1">{profile?.email ?? "desktop@lune.local"}</p>
            <div className="sidebar-list mt-3">
              <div className="sidebar-list-row">
                <span className="sidebar-key">Recording</span>
                <span className="sidebar-value">Ready now</span>
              </div>
              <div className="sidebar-list-row">
                <span className="sidebar-key">Vocabulary</span>
                <span className="sidebar-value">Saved fixes</span>
              </div>
              <div className="sidebar-list-row">
                <span className="sidebar-key">Shortcuts</span>
                <span className="sidebar-value">Across apps</span>
              </div>
            </div>
            <Button
              type="button"
              variant="ghost"
              className="mt-4 w-full justify-center"
              onClick={authActions.signOut}
            >
              Sign out
            </Button>
          </div>
        ) : null}

        <Button
          type="button"
          variant="sidebar"
          onClick={workspaceActions.toggleSidebar}
        >
          {collapsed ? <ChevronRightIcon className="h-4 w-4" /> : <ChevronLeftIcon className="h-4 w-4" />}
          {!collapsed ? (
            <div className="text-left">
              <div className="supporting-label text-copy-50">Collapse rail</div>
              <div className="sidebar-card-copy">Compact the workspace chrome</div>
            </div>
          ) : null}
        </Button>
      </div>
    </aside>
  );
}
