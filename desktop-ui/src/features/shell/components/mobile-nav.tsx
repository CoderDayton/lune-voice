import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { cn } from "../../../lib/cn";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { navigationItems } from "../config/navigation";
import { CloseIcon, LuneLogo } from "../config/icons";

export function MobileNav() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const open = useStore(workspaceStore, (state) => state.mobileMenuOpen);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 md:hidden">
      {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        tabIndex={-1}
        ref={(el) => el?.focus()}
        onClick={workspaceActions.closeMobileMenu}
        onKeyDown={(e) => e.key === "Escape" && workspaceActions.closeMobileMenu()}
      />

      <nav className="sidebar-surface absolute inset-y-0 left-0 flex w-72 flex-col gap-4 py-4 px-3 overflow-y-auto">
        <div className="flex items-center justify-between px-2">
          <div className="flex items-center gap-3">
            <div className="brand-mark">
              <LuneLogo className="text-white" />
            </div>
            <div className="min-w-0">
              <p className="app-kicker">Desktop Dictation</p>
              <h2 className="brand-title">Lune Voice</h2>
            </div>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            aria-label="Close navigation menu"
            onClick={workspaceActions.closeMobileMenu}
          >
            <CloseIcon className="size-5" />
          </button>
        </div>

        <div className="divider my-0" />

        <div className="flex flex-col gap-1">
          {navigationItems.map(({ to, label, caption, icon: Icon }) => {
            const active = to === "/" ? pathname === "/" : pathname.startsWith(to);

            return (
              <Link
                key={to}
                to={to}
                className={cn(active ? "nav-link-active" : "nav-link-idle")}
                onClick={workspaceActions.closeMobileMenu}
              >
                <Icon className={cn("nav-icon", active ? "text-secondary" : "opacity-50")} />
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">{label}</div>
                  <div className="truncate text-xs opacity-50">{caption}</div>
                </div>
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
