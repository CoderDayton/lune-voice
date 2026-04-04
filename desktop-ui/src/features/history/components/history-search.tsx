import { useStore } from "@tanstack/react-store";
import { Search } from "lucide-react";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function HistorySearch() {
  const { t } = useLingui();
  const search = useStore(workspaceStore, (state) => state.historySearch);

  return (
    <label className="input w-full">
      <Search size={16} strokeWidth={2} className="opacity-40" />
      <input
        type="search"
        value={search}
        onChange={(e) => workspaceActions.setHistorySearch(e.target.value)}
        placeholder={t`Search transcripts…`}
        className="grow"
      />
    </label>
  );
}
