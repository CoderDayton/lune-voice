import { useStore } from "@tanstack/react-store";
import { HistorySearch } from "./components/history-search";
import { HistorySessionList } from "./components/history-session-list";
import { getVisibleHistorySessions, workspaceStore } from "../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function HistoryPage() {
  const { t } = useLingui();
  const count = useStore(workspaceStore, (s) => getVisibleHistorySessions(s).length);

  return (
    <div className="page-stack">
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <HistorySearch />
        </div>
        <span className="text-sm opacity-50 tabular-nums shrink-0">
          {count} {t`sessions`}
        </span>
      </div>
      <HistorySessionList />
    </div>
  );
}
