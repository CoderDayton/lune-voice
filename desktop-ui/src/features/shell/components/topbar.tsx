import { useStore } from "@tanstack/react-store";
import { Button } from "../../../components/ui/button";
import { Pill } from "../../../components/ui/pill";
import {
  cancelDictation,
  startDictation,
  stopDictation,
  workspaceActions,
  workspaceStore,
} from "../../../store/workspace-store";
import type { NavigationItem } from "../config/navigation";
import { MenuIcon } from "../config/icons";

interface TopbarProps {
  page: NavigationItem;
}

export function Topbar({ page }: TopbarProps) {
  const runtime = useStore(workspaceStore, (state) => state.runtime);
  const isActive = runtime.activeSessionId !== null || runtime.serviceStatus === "listening";
  const isBusy = runtime.loading || runtime.serviceStatus === "processing";

  return (
    <header className="topbar-shell">
      <div className="flex items-start gap-3">
        <button
          type="button"
          className="btn btn-ghost btn-sm md:hidden mt-1"
          aria-label="Open navigation menu"
          onClick={workspaceActions.toggleMobileMenu}
        >
          <MenuIcon className="size-5" />
        </button>
        <div className="topbar-copy">
          <p className="app-kicker">{page.eyebrow}</p>
          <h1 className="page-title mt-2">{page.label}</h1>
          <p className="body-copy-soft mt-2">{page.description}</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Pill>Private on-device</Pill>
        <Pill>{runtime.serviceStatus}</Pill>
        <Pill>{runtime.ipcFlavor}</Pill>
        <Button variant="secondary" disabled={isActive || isBusy} onClick={() => void startDictation()}>
          Start
        </Button>
        <Button variant="ghost" disabled={!isActive} onClick={() => void stopDictation()}>
          Stop
        </Button>
        <Button variant="danger" disabled={!isActive} onClick={() => void cancelDictation()}>
          Cancel
        </Button>
      </div>
    </header>
  );
}
