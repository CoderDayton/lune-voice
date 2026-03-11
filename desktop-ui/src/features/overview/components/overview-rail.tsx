import { Link } from "@tanstack/react-router";
import { buttonClassName } from "../../../components/ui/button";
import { Panel } from "../../../components/ui/panel";
import { overviewSignals } from "../data/overview-data";

export function OverviewRail() {
  return (
    <div className="stack-section">
      <Panel className="panel-pad-lg">
        <p className="app-kicker-accent">Workspace posture</p>
        <div className="rail-list mt-4">
          {overviewSignals.map((item) => (
            <div key={item.title} className="rail-row">
              <div className="rail-copy">
                <div className="supporting-label text-copy-50">{item.title}</div>
                <div className="body-copy-soft mt-1">{item.note}</div>
              </div>
              <div className="rail-value">{item.value}</div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel className="panel-pad-lg">
        <p className="app-kicker-accent">Next move</p>
        <h3 className="section-title mt-3">Normalize repeated corrections.</h3>
        <p className="body-copy-soft mt-2">
          The fastest visible quality gain now is tightening the dictionary layer around common house terms and code nouns.
        </p>
        <div className="mt-5">
          <Link to="/dictionary" className={buttonClassName("primary")}>
            Open dictionary
          </Link>
        </div>
      </Panel>
    </div>
  );
}
