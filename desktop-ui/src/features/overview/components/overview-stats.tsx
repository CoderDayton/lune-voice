import { BoltIcon, ClockIcon, MicIcon, PulseIcon } from "../../shell/config/icons";
import { MetricCard } from "../../../components/ui/metric-card";
import { overviewStats } from "../data/overview-data";

const icons = [BoltIcon, ClockIcon, MicIcon, PulseIcon];

export function OverviewStats() {
  return (
    <section className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
      {overviewStats.map((item, index) => {
        const Icon = icons[index];
        return <MetricCard key={item.label} label={item.label} value={item.value} note={item.note} icon={<Icon className="h-4 w-4" />} />;
      })}
    </section>
  );
}
