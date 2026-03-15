import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { Button } from "../../../components/ui/button";
import { Pill } from "../../../components/ui/pill";
import { cn } from "../../../lib/cn";
import { authActions, authStore } from "../../../store/auth-store";
import {
  summarizeAsrBackend,
  summarizeDictationMode,
  workspaceActions,
  workspaceStore,
} from "../../../store/workspace-store";
import { findNavigationItem, navigationItems } from "../config/navigation";
import { ChevronLeftIcon, ChevronRightIcon, LuneLogo } from "../config/icons";
import { Trans } from "@lingui/react/macro";

function useStatusItems() {
  const runtime = useStore(workspaceStore, (state) => state.runtime);
  const settings = useStore(workspaceStore, (state) => state.settings);

  const statusLabels: Record<string, string> = {
    ready: "Ready", listening: "Listening", processing: "Processing", error: "Error",
  };
  const serviceLabel = statusLabels[runtime.serviceStatus] ?? "Connecting";

  const serviceTone =
    runtime.serviceStatus === "error"
      ? "text-error"
      : runtime.serviceStatus === "listening"
        ? "text-accent"
        : "text-success";

  return [
    { label: "Service", value: serviceLabel, tone: serviceTone },
    { label: "Speech model", value: summarizeAsrBackend(settings.asrBackend), tone: "text-secondary" },
    { label: "Delivery", value: summarizeDictationMode(settings.dictationMode), tone: "text-primary" },
  ];
}

export function Sidebar() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const collapsed = useStore(workspaceStore, (state) => state.sidebarCollapsed);
  const profile = useStore(authStore, (state) => state.profile);
  const current = findNavigationItem(pathname);
  const statusItems = useStatusItems();
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
      className="sidebar-surface relative z-10 hidden md:flex shrink-0 flex-col gap-4 py-4 px-3 overflow-y-auto transition-[width] duration-200"
      style={{
        width: collapsed ? "var(--size-sidebar-collapsed)" : "var(--size-sidebar)",
      }}
    >
      <div className="flex items-center gap-3 px-2">
        <div className="brand-mark">
          <LuneLogo className="text-white" />
        </div>
        {!collapsed ? (
          <div className="min-w-0">
            <p className="app-kicker"><Trans>Desktop Dictation</Trans></p>
            <h1 className="brand-title"><Trans>Lune Voice</Trans></h1>
          </div>
        ) : null}
      </div>

      <div className="divider my-0" />

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
                  <Icon className={cn("nav-icon", active ? "text-secondary" : "opacity-50")} />
                  {!collapsed ? (
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium">{label}</div>
                      <div className="truncate text-xs opacity-50">{caption}</div>
                    </div>
                  ) : null}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {!collapsed ? (
        <div className="card bg-base-300/50 border border-base-300/30 p-4">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div>
              <p className="app-kicker">Session posture</p>
              <h2 className="font-semibold text-sm mt-2"><Trans>{current.label} is ready</Trans></h2>
              <p className="text-xs opacity-50 mt-1">
                <Trans>Your recordings stay private on this computer, with quick cleanup before text is inserted.</Trans>
              </p>
            </div>
            <Pill>{statusItems[0].value}</Pill>
          </div>

          <div className="flex flex-col gap-3">
            {statusItems.map((item) => (
              <div key={item.label} className="flex items-center justify-between gap-3 text-sm">
                <span className="opacity-50">{item.label}</span>
                <span className={cn("font-semibold", item.tone)}>{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="mt-auto space-y-3">
        {!collapsed ? (
          <div className="card bg-base-300/50 border border-base-300/30 p-4">
            <p className="app-kicker"><Trans>Signed in</Trans></p>
            <h2 className="font-semibold text-sm mt-2">{profile?.name ?? <Trans>Lune user</Trans>}</h2>
            <p className="text-xs opacity-50 mt-1">{profile?.email ?? "desktop@lune.local"}</p>
            <div className="flex flex-col gap-3 mt-3">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="opacity-50"><Trans>Recording</Trans></span>
                <span className="font-medium opacity-80"><Trans>Ready now</Trans></span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="opacity-50"><Trans>Vocabulary</Trans></span>
                <span className="font-medium opacity-80"><Trans>Saved fixes</Trans></span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="opacity-50"><Trans>Shortcuts</Trans></span>
                <span className="font-medium opacity-80"><Trans>Across apps</Trans></span>
              </div>
            </div>
            <Button
              type="button"
              variant="ghost"
              className="mt-4 w-full justify-center"
              onClick={authActions.signOut}
            >
              <Trans>Sign out</Trans>
            </Button>
          </div>
        ) : null}

        <Button
          type="button"
          variant="sidebar"
          onClick={workspaceActions.toggleSidebar}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRightIcon className="h-4 w-4" /> : <ChevronLeftIcon className="h-4 w-4" />}
          {!collapsed ? (
            <div className="text-left">
              <div className="text-sm font-medium"><Trans>Collapse rail</Trans></div>
              <div className="text-xs opacity-50"><Trans>Compact the workspace chrome</Trans></div>
            </div>
          ) : null}
        </Button>
      </div>
    </aside>
  );
}
