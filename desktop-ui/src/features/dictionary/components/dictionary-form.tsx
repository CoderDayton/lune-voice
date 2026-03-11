import { useState } from "react";
import { Button } from "../../../components/ui/button";
import { TextField } from "../../../components/ui/field";
import { Panel } from "../../../components/ui/panel";
import { addDictionaryEntry } from "../../../store/workspace-store";

export function DictionaryForm() {
  const [phrase, setPhrase] = useState("");
  const [replacement, setReplacement] = useState("");

  return (
    <Panel className="panel-pad-lg">
      <p className="app-kicker-accent">Add rule</p>
      <h3 className="section-title mt-3">Teach the correction layer your vocabulary.</h3>
      <p className="body-copy-soft mt-2">
        Normalize repeated product names, acronyms, and house terminology before text is delivered.
      </p>

      <div className="layout-grid mt-6 lg:grid-cols-[1fr_auto_1fr_auto] lg:items-end">
        <div>
          <label className="field-label">Spoken phrase</label>
          <TextField value={phrase} onChange={(event) => setPhrase(event.target.value)} placeholder="e.g. github" />
        </div>
        <div className="hidden pb-3 section-title text-copy-500 lg:block">→</div>
        <div>
          <label className="field-label">Replacement</label>
          <TextField value={replacement} onChange={(event) => setReplacement(event.target.value)} placeholder="e.g. GitHub" />
        </div>
        <Button
          onClick={async () => {
            const added = await addDictionaryEntry(phrase, replacement);

            if (added) {
              setPhrase("");
              setReplacement("");
            }
          }}
        >
          Add rule
        </Button>
      </div>
    </Panel>
  );
}
