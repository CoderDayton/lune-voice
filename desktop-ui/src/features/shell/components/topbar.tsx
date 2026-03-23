import { useRouterState } from "@tanstack/react-router";
import { workspaceActions } from "../../../store/workspace-store";
import { findNavigationItem } from "../config/navigation";
import { MenuIcon } from "../config/icons";
import { useLingui } from "@lingui/react/macro";

export function Topbar() {
  const { t } = useLingui();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const page = findNavigationItem(pathname);

  return (
    <div className="navbar px-4 sm:px-6">
      <div className="navbar-start">
        <button
          type="button"
          className="btn btn-ghost btn-sm md:hidden"
          aria-label={t`Open navigation menu`}
          onClick={workspaceActions.toggleMobileMenu}
        >
          <MenuIcon className="size-5" />
        </button>
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight select-none">{page.label}</h1>
      </div>
    </div>
  );
}
