import { useStore } from "@tanstack/react-store";
import { Link, useRouterState } from "@tanstack/react-router";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { navigationItems } from "../config/navigation";
import { CloseIcon } from "../config/icons";
import luneLogo from "../../../assets/lune-voice-logo.png";
import { useLingui } from "@lingui/react/macro";

export function MobileNav() {
  const { t } = useLingui();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const open = useStore(workspaceStore, (state) => state.mobileMenuOpen);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 md:hidden">
      <button
        type="button"
        className="absolute inset-0 bg-black/60 appearance-none border-none cursor-default"
        aria-label={t`Close navigation`}
        onClick={workspaceActions.closeMobileMenu}
        onKeyDown={(e) => e.key === "Escape" && workspaceActions.closeMobileMenu()}
      />

      <nav className="sidebar-surface absolute inset-y-0 left-0 flex w-64 flex-col gap-6 py-4 px-3 overflow-y-auto">
        <div className="flex items-center justify-between px-2">
          <div className="flex items-center gap-3">
            <div className="size-10 shrink-0 select-none pointer-events-none" aria-hidden="true">
              <div className="size-full rounded-xl overflow-hidden">
                <img src={luneLogo} alt="" className="size-full object-cover" draggable={false} />
              </div>
            </div>
            <span className="brand-title">{t`Lune Voice`}</span>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            aria-label={t`Close navigation menu`}
            onClick={workspaceActions.closeMobileMenu}
          >
            <CloseIcon className="size-5" />
          </button>
        </div>

        <ul className="menu gap-1 px-0">
          {navigationItems.map(({ to, label, icon: Icon }) => {
            const active = to === "/" ? pathname === "/" : pathname.startsWith(to);
            return (
              <li key={to}>
                <Link
                  to={to}
                  className={active ? "menu-active" : ""}
                  onClick={workspaceActions.closeMobileMenu}
                >
                  <Icon className={`nav-icon${active ? " text-secondary" : ""}`} />
                  <span className="truncate">{label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}
