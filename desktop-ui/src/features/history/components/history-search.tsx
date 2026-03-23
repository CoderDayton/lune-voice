import { useStore } from "@tanstack/react-store";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function HistorySearch() {
  const { t } = useLingui();
  const search = useStore(workspaceStore, (state) => state.historySearch);

  return (
    <label className="input w-full">
      <svg className="h-4 w-4 opacity-50" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
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
