import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { MetricCard } from "../../../components/ui/metric-card";
import { historyGroups } from "../data/history-data";
import { ClockIcon, MicIcon, PulseIcon, SearchIcon } from "../../shell/config/icons";
import { workspaceStore } from "../../../store/workspace-store";

export function HistorySummary() {
  const search = useStore(workspaceStore, (state) => state.historySearch);

  const summary = useMemo(() => {
    const items = historyGroups.flatMap((group) => group.items);
    const visible = items.filter((item) => !search || item.text.toLowerCase().includes(search.toLowerCase()));
    const totalWords = visible.reduce((sum, item) => sum + item.words, 0);
    const totalSeconds = visible.reduce((sum, item) => sum + item.seconds, 0);
    const modes = new Set(visible.map((item) => item.mode));

    let source = "No sessions";

    if (modes.size === 1) {
      source = visible[0]?.mode ?? source;
    } else if (modes.size > 1) {
      source = "Mixed";
    }

    return {
      sessions: String(visible.length),
      words: totalWords.toLocaleString(),
      duration: `${Math.floor(totalSeconds / 60)}m ${String(totalSeconds % 60).padStart(2, "0")}s`,
      source,
    };
  }, [search]);

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="Visible sessions" value={summary.sessions} note="Records in the active search slice." icon={<SearchIcon className="h-4 w-4" />} />
      <MetricCard label="Visible words" value={summary.words} note="Total dictated words in the active archive view." icon={<MicIcon className="h-4 w-4" />} />
      <MetricCard label="Speaking time" value={summary.duration} note="Combined audio duration from the visible records." icon={<ClockIcon className="h-4 w-4" />} />
      <MetricCard label="Voice mode" value={summary.source} note="How the visible sessions were captured and cleaned up." icon={<PulseIcon className="h-4 w-4" />} />
    </div>
  );
}
