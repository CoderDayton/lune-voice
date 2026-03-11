import { Panel } from "../../components/ui/panel";
import { DictionaryForm } from "./components/dictionary-form";
import { DictionaryRail } from "./components/dictionary-rail";
import { DictionaryTable } from "./components/dictionary-table";

export function DictionaryPage() {
  return (
    <div className="page-stack">
      <Panel className="panel-pad-lg">
        <div className="page-intro">
          <p className="app-kicker-accent">Vocabulary</p>
          <h2 className="page-title mt-3">Save the names and phrases you want Lune Voice to get right.</h2>
          <p className="body-copy mt-3">
            Add common names, acronyms, and preferred spelling so repeated corrections happen automatically.
          </p>
        </div>
      </Panel>

      <DictionaryForm />

      <section className="page-grid-dictionary">
        <DictionaryTable />
        <DictionaryRail />
      </section>
    </div>
  );
}
