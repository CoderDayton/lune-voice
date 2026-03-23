import { useState } from "react";
import { addDictionaryEntry } from "../../../store/workspace-store";
import { Trans, useLingui } from "@lingui/react/macro";

export function DictionaryForm() {
  const { t } = useLingui();
  const [phrase, setPhrase] = useState("");
  const [replacement, setReplacement] = useState("");

  async function handleSubmit() {
    const added = await addDictionaryEntry(phrase, replacement);
    if (added) {
      setPhrase("");
      setReplacement("");
    }
  }

  return (
    <fieldset className="fieldset bg-base-200 border-base-300 rounded-box border p-4">
      <legend className="fieldset-legend"><Trans>Add correction rule</Trans></legend>
      <div className="flex flex-col sm:flex-row gap-3 items-end">
        <label className="form-control flex-1">
          <div className="label"><span className="label-text"><Trans>Spoken phrase</Trans></span></div>
          <input
            type="text"
            className="input w-full"
            value={phrase}
            onChange={(e) => setPhrase(e.target.value)}
            placeholder={t`e.g. github…`}
          />
        </label>
        <span className="hidden sm:block pb-3 opacity-30 text-lg">→</span>
        <label className="form-control flex-1">
          <div className="label"><span className="label-text"><Trans>Replacement</Trans></span></div>
          <input
            type="text"
            className="input w-full"
            value={replacement}
            onChange={(e) => setReplacement(e.target.value)}
            placeholder={t`e.g. GitHub…`}
          />
        </label>
        <button type="button" className="btn btn-primary" onClick={handleSubmit}>
          <Trans>Add</Trans>
        </button>
      </div>
    </fieldset>
  );
}
