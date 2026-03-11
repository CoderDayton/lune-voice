import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { HistorySearch } from "./history-search";

export function HistoryRail() {
  return (
    <div className="stack-section">
      <Panel className="panel-pad-md">
        <p className="app-kicker-accent">Archive filter</p>
        <h3 className="card-title mt-3">Search the local transcript log.</h3>
        <p className="body-copy-soft mt-2">
          Filter across saved sessions to spot regressions, repeated prompts, or correction misses.
        </p>
        <div className="mt-5">
          <HistorySearch />
        </div>
      </Panel>

      <Panel className="panel-pad-md">
        <p className="app-kicker-accent">Archive signals</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Pill>Offline storage</Pill>
          <Pill>Benchmark traces</Pill>
          <Pill>Correction review</Pill>
        </div>
        <div className="insight-stack mt-5">
          <div className="insight-block">
            <div className="label-text">Retention</div>
            <div className="body-copy-strong mt-2">Keep the latest sessions visible while keeping long-term notes searchable.</div>
          </div>
          <div className="insight-block">
            <div className="label-text">Quality loop</div>
            <div className="body-copy-strong mt-2">Use this view to move repeated misses into the dictionary and confirm benchmark changes.</div>
          </div>
        </div>
      </Panel>
    </div>
  );
}
