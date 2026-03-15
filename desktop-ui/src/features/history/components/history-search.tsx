import { useStore } from "@tanstack/react-store";
import { TextField } from "../../../components/ui/field";
import { SearchIcon } from "../../shell/config/icons";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function HistorySearch() {
  const { t } = useLingui();
  const search = useStore(workspaceStore, (state) => state.historySearch);

  return (
    <label className="relative block">
      <SearchIcon className="search-icon" />
      <TextField
        value={search}
        onChange={(event) => workspaceActions.setHistorySearch(event.target.value)}
        aria-label={t`Search transcripts`}
        placeholder={t`Search transcripts…`}
        className="field-with-icon"
      />
    </label>
  );
}
