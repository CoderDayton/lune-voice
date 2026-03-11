import { useStore } from "@tanstack/react-store";
import { TextField } from "../../../components/ui/field";
import { SearchIcon } from "../../shell/config/icons";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";

export function HistorySearch() {
  const search = useStore(workspaceStore, (state) => state.historySearch);

  return (
    <label className="relative block">
      <SearchIcon className="search-icon" />
      <TextField
        value={search}
        onChange={(event) => workspaceActions.setHistorySearch(event.target.value)}
        placeholder="Search transcripts"
        className="field-with-icon"
      />
    </label>
  );
}
