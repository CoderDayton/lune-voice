import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { ChevronsLeft, ChevronsRight } from "lucide-react";
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
      {/* Brand */}
      <div className="flex w-full items-center gap-3 px-4 py-5 select-none is-drawer-close:justify-center is-drawer-close:px-2 sm:gap-3.5 sm:px-5 sm:py-6">
        <div className="relative shrink-0">
          <div className="size-10 rounded-xl ring-1 ring-white/10 overflow-hidden">
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
            <span className="font-light text-base-content/60"> Voice</span>
          </span>
        </div>
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
                  className={`group relative flex w-full items-center gap-3.5 rounded-xl px-3.5 py-4 text-[0.9rem] transition-colors is-drawer-close:tooltip is-drawer-close:tooltip-right is-drawer-close:justify-center ${active ? "bg-primary/10 text-primary font-medium" : "text-base-content/55 hover:bg-base-content/[0.04] hover:text-base-content/90"}`}
                  data-tip={label}
                >
                  {active ? (
                    <span className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-1 rounded-full bg-primary" />
                  ) : null}
                  <Icon
                    size={20}
                    strokeWidth={active ? 2.2 : 1.6}
                    className={active ? "text-primary" : "text-base-content/45 group-hover:text-base-content/70"}
                  />
                  <span className="truncate is-drawer-close:hidden">{label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="flex-1" />

      <div className="is-drawer-close:hidden">
        <SidebarQuickStart />
        <SidebarTodayStats />
      </div>

      {/* Footer */}
      <div className="mx-3.5 border-t border-base-content/5" />
      <div className="flex items-center gap-1 px-3 py-3">
        <label
          htmlFor={DRAWER_ID}
          className="flex cursor-pointer items-center gap-3.5 rounded-xl px-3.5 py-3 text-sm transition-colors text-base-content/40 hover:bg-base-content/[0.04] hover:text-base-content/70 is-drawer-close:w-full is-drawer-close:justify-center"
          aria-label={t`Toggle sidebar`}
        >
          <ChevronsLeft size={20} strokeWidth={1.5} className="is-drawer-close:hidden" />
          <ChevronsRight size={20} strokeWidth={1.5} className="is-drawer-open:hidden" />
          <span className="truncate is-drawer-close:hidden">{t`Collapse`}</span>
        </label>
        <span className="ml-auto pr-1 text-xs tabular-nums text-base-content/20 select-none is-drawer-close:hidden">v0.1</span>
      </div>
    </>
  );
}
