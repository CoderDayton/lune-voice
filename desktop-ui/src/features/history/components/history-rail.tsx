import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { Trans } from "@lingui/react/macro";
import { HistorySearch } from "./history-search";

export function HistoryRail() {
  return (
    <div className="stack-section">
      <Panel className="p-4 sm:p-5">
        <p className="app-kicker-accent"><Trans>Archive filter</Trans></p>
        <h3 className="card-title-text mt-3"><Trans>Search the local transcript log.</Trans></h3>
        <p className="body-copy-soft mt-2">
          <Trans>Filter across saved sessions to spot regressions, repeated prompts, or correction misses.</Trans>
        </p>
        <div className="mt-5">
          <HistorySearch />
        </div>
      </Panel>

      <Panel className="p-4 sm:p-5">
        <p className="app-kicker-accent"><Trans>Archive signals</Trans></p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Pill><Trans>Offline storage</Trans></Pill>
          <Pill><Trans>Benchmark traces</Trans></Pill>
          <Pill><Trans>Correction review</Trans></Pill>
        </div>
        <div className="flex flex-col gap-4 mt-5">
          <div className="stack-copy-tight">
            <div className="label-text-custom"><Trans>Retention</Trans></div>
            <div className="body-copy-strong mt-2"><Trans>Keep the latest sessions visible while keeping long-term notes searchable.</Trans></div>
          </div>
          <div className="stack-copy-tight">
            <div className="label-text-custom"><Trans>Quality loop</Trans></div>
            <div className="body-copy-strong mt-2"><Trans>Use this view to move repeated misses into the dictionary and confirm benchmark changes.</Trans></div>
          </div>
        </div>
      </Panel>
    </div>
  );
}
