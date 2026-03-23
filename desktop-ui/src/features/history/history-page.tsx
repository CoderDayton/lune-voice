import { useStore } from "@tanstack/react-store";
import { HistorySearch } from "./components/history-search";
import { HistorySessionList } from "./components/history-session-list";
import { workspaceStore } from "../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

function selectSessionCount(s: { historySessions: { id: string }[]; historySearch: string }) {
  const query = s.historySearch.trim().toLowerCase();
  if (!query) return s.historySessions.length;
  return s.historySessions.filter((sess) => "transcript" in sess && String((sess as { transcript: string }).transcript).toLowerCase().includes(query)).length;
}

export function HistoryPage() {
  const { t } = useLingui();
  const count = useStore(workspaceStore, selectSessionCount);

  return (
    <div className="page-stack">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
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
