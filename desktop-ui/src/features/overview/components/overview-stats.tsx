import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { BoltIcon, ClockIcon, MicIcon, PulseIcon } from "../../shell/config/icons";
import { MetricCard } from "../../../components/ui/metric-card";
import { workspaceStore } from "../../../store/workspace-store";

const icons = [BoltIcon, ClockIcon, MicIcon, PulseIcon];

export function OverviewStats() {
  const historySessions = useStore(workspaceStore, (state) => state.historySessions);
  const runtime = useStore(workspaceStore, (state) => state.runtime);

  const stats = useMemo(() => {
    const totalWords = historySessions.reduce((sum, session) => sum + session.words, 0);
    const totalSeconds = historySessions.reduce((sum, session) => sum + session.seconds, 0);
    const averagePace = totalSeconds > 0 ? Math.round((totalWords / totalSeconds) * 60) : 0;

    return [
      { label: "Total words", value: totalWords.toLocaleString(), note: "Words transcribed from the current local archive." },
      { label: "Speaking time", value: formatDuration(totalSeconds), note: "Total captured duration across recent working sessions." },
      { label: "Sessions", value: historySessions.length.toString(), note: "Finished recordings stored only on this device." },
      { label: "IPC layer", value: runtime.ipcFlavor, note: "Current frontend-to-backend connection mode." },
      { label: "Avg pace", value: `${averagePace} wpm`, note: "Measured across the currently loaded local sessions." },
    ];
  }, [historySessions, runtime.ipcFlavor]);

  return (
    <section className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
      {stats.slice(0, 4).map((item, index) => {
        const Icon = icons[index] ?? PulseIcon;
        return <MetricCard key={item.label} label={item.label} value={item.value} note={item.note} icon={<Icon className="h-4 w-4" />} />;
      })}
    </section>
  );
}

function formatDuration(totalSeconds: number) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}
