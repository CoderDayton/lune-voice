import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { workspaceStore } from "../../../store/workspace-store";
import { navigationItems } from "../config/navigation";
import { SidebarQuickStart, SidebarTodayStats } from "./sidebar-widgets";
import { DRAWER_ID } from "./app-shell";
import luneLogo from "../../../assets/lune-voice-logo.png";
import { useLingui } from "@lingui/react/macro";

export function SidebarContent() {
  const { t } = useLingui();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);

  const isActive = serviceStatus === "listening" || serviceStatus === "processing";
  const isConnected = serviceStatus === "ready" || isActive;

  return (
    <>
      {/* Brand + collapse toggle */}
      <div className="flex w-full items-center gap-3 px-4 py-5 select-none is-drawer-close:justify-center is-drawer-close:px-2 sm:gap-3.5 sm:px-5 sm:py-6">
        <div className="relative shrink-0">
          <div className="size-10 rounded-field ring-1 ring-white/10 overflow-hidden">
            <img src={luneLogo} alt="" className="size-full object-cover pointer-events-none" draggable={false} />
          </div>
          <span
            className={`absolute -bottom-0.5 -right-0.5 block size-2.5 rounded-full ring-2 ring-base-100 ${isActive ? "bg-success status-pulse" : isConnected ? "bg-success/50" : "bg-base-content/20"}`}
            aria-hidden="true"
          />
        </div>
        <div className="min-w-0 flex-1 is-drawer-close:hidden">
          <span className="block text-lg leading-tight tracking-tight">
            <span className="font-bold">Lune</span>
            <span className="font-light text-content-muted"> Voice</span>
          </span>
        </div>
        <label
          htmlFor={DRAWER_ID}
          className="ml-auto cursor-pointer text-content-muted transition-colors hover:text-content is-drawer-close:hidden"
          aria-label={t`Collapse sidebar`}
        >
          <PanelLeftClose size={20} strokeWidth={2} />
        </label>
      </div>

      {/* Navigation */}
      <nav className="px-3 py-2">
        <p className="section-label mb-3 px-3.5 is-drawer-close:hidden">{t`Navigate`}</p>
        <ul className="flex flex-col gap-1">
          {navigationItems.map(({ to, label, icon: Icon }) => {
            const active = to === "/" ? pathname === "/" : pathname.startsWith(to);
            return (
              <li key={to}>
                <Link
                  to={to}
                  aria-current={active ? "page" : undefined}
                  className={`group relative flex w-full items-center gap-3.5 rounded-field px-3.5 py-4 text-[0.9rem] transition-colors is-drawer-close:tooltip is-drawer-close:tooltip-right is-drawer-close:justify-center ${active ? "bg-primary/10 text-primary font-medium" : "text-content-muted hover:bg-surface-hover hover:text-content"}`}
                  data-tip={label}
                >
                  {active ? (
                    <span className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-1 rounded-full bg-primary" />
                  ) : null}
                  <Icon
                    size={20}
                    strokeWidth={active ? 2.2 : 1.6}
                    className={active ? "text-primary" : "text-content-subtle group-hover:text-content-muted"}
                  />
                  <span className="truncate is-drawer-close:hidden">{label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="flex-1" />

      {/* Expand button — bottom center, only when collapsed */}
      <div className="is-drawer-open:hidden flex justify-center mb-6">
        <label
          htmlFor={DRAWER_ID}
          className="cursor-pointer text-content-muted transition-colors hover:text-content"
          aria-label={t`Expand sidebar`}
        >
          <PanelLeftOpen size={26} strokeWidth={2} />
        </label>
      </div>

      <div className="is-drawer-close:hidden">
        <SidebarQuickStart />
        <SidebarTodayStats />
      </div>
    </>
  );
}
