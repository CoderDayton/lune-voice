import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { MetricCard } from "../../../components/ui/metric-card";
import { ClockIcon, MicIcon, PulseIcon, SearchIcon } from "../../shell/config/icons";
import {
  formatBackendLabel,
  getVisibleHistorySessions,
  workspaceStore,
} from "../../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function HistorySummary() {
  const { t } = useLingui();
  const visible = useStore(workspaceStore, getVisibleHistorySessions);

  const summary = useMemo(() => {
    const totalWords = visible.reduce((sum, item) => sum + item.words, 0);
    const totalSeconds = visible.reduce((sum, item) => sum + item.seconds, 0);
    const modes = new Set(visible.map((item) => item.backend));

    let source = "No sessions";

    if (modes.size === 1) {
      source = visible[0] ? formatBackendLabel(visible[0].backend) : source;
    } else if (modes.size > 1) {
      source = "Mixed";
    }

    return {
      sessions: String(visible.length),
      words: totalWords.toLocaleString(),
      duration: `${Math.floor(totalSeconds / 60)}m ${String(totalSeconds % 60).padStart(2, "0")}s`,
      source,
    };
  }, [visible]);

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label={t`Visible sessions`} value={summary.sessions} note={t`Records in the active search slice.`} icon={<SearchIcon className="h-4 w-4" />} />
      <MetricCard label={t`Visible words`} value={summary.words} note={t`Total dictated words in the active archive view.`} icon={<MicIcon className="h-4 w-4" />} />
      <MetricCard label={t`Speaking time`} value={summary.duration} note={t`Combined audio duration from the visible records.`} icon={<ClockIcon className="h-4 w-4" />} />
      <MetricCard label={t`Voice mode`} value={summary.source} note={t`How the visible sessions were captured and cleaned up.`} icon={<PulseIcon className="h-4 w-4" />} />
    </div>
  );
}
