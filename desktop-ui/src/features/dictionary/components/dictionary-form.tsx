import { useState } from "react";
import { Button } from "../../../components/ui/button";
import { TextField } from "../../../components/ui/field";
import { Panel } from "../../../components/ui/panel";
import { addDictionaryEntry } from "../../../store/workspace-store";
import { Trans, useLingui } from "@lingui/react/macro";

export function DictionaryForm() {
  const { t } = useLingui();
  const [phrase, setPhrase] = useState("");
  const [replacement, setReplacement] = useState("");

  return (
    <Panel className="p-4 sm:p-6">
      <p className="app-kicker-accent"><Trans>Add rule</Trans></p>
      <h3 className="section-title mt-3"><Trans>Teach the correction layer your vocabulary.</Trans></h3>
      <p className="body-copy-soft mt-2">
        <Trans>Normalize repeated product names, acronyms, and house terminology before text is delivered.</Trans>
      </p>

      <div className="layout-grid mt-6 lg:grid-cols-[1fr_auto_1fr_auto] lg:items-end">
        <div>
          <label className="label-text-custom block mb-2"><Trans>Spoken phrase</Trans></label>
          <TextField value={phrase} onChange={(event) => setPhrase(event.target.value)} placeholder={t`e.g. github…`} />
        </div>
        <div className="hidden pb-3 section-title opacity-50 lg:block">→</div>
        <div>
          <label className="label-text-custom block mb-2"><Trans>Replacement</Trans></label>
          <TextField value={replacement} onChange={(event) => setReplacement(event.target.value)} placeholder={t`e.g. GitHub…`} />
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
          <Trans>Add rule</Trans>
        </Button>
      </div>
    </Panel>
  );
}
