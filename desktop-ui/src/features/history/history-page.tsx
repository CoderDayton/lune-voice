import { useStore } from "@tanstack/react-store";
import { HistorySearch } from "./components/history-search";
import { HistorySessionList } from "./components/history-session-list";
import { getVisibleSessionCount, workspaceStore } from "../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function HistoryPage() {
  const { t } = useLingui();
  const count = useStore(workspaceStore, getVisibleSessionCount);

  return (
    <div className="page-stack">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
        <div className="flex-1">
          <HistorySearch />
        </div>
        <span className="text-sm text-content-subtle tabular-nums shrink-0">
          {count} {t`sessions`}
        </span>
      </div>
      <HistorySessionList />
    </div>
  );
}
