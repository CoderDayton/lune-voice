import { useActionState } from "react";
import { addDictionaryEntry } from "../../../store/workspace-store";
import { Trans, useLingui } from "@lingui/react/macro";

interface FormState {
  phrase: string;
  replacement: string;
  error: string | null;
}

const initialState: FormState = { phrase: "", replacement: "", error: null };

export function DictionaryForm() {
  const { t } = useLingui();

  const [state, submitAction, isPending] = useActionState(
    async (_prev: FormState, formData: FormData) => {
      const phrase = (formData.get("phrase") as string) ?? "";
      const replacement = (formData.get("replacement") as string) ?? "";
      const added = await addDictionaryEntry(phrase, replacement);
      if (added) return initialState;
      return { phrase, replacement, error: "Failed to add entry" };
    },
    initialState,
  );

  return (
    <form action={submitAction}>
      <section className="card bg-base-200 border border-base-300/30 rounded-box p-5 sm:p-6">
        <h3 className="text-lg font-semibold tracking-tight mb-4"><Trans>Add correction rule</Trans></h3>
        <fieldset className="flex flex-col sm:flex-row gap-3 items-end" disabled={isPending}>
          <label className="form-control flex-1">
            <div className="label"><span className="label-text"><Trans>Spoken phrase</Trans></span></div>
            <input
              type="text"
              name="phrase"
              className="input w-full"
              defaultValue={state.phrase}
              key={state === initialState ? "reset-phrase" : undefined}
              placeholder={t`e.g. github…`}
            />
          </label>
          <span className="hidden sm:block pb-3 opacity-30 text-lg">→</span>
          <label className="form-control flex-1">
            <div className="label"><span className="label-text"><Trans>Replacement</Trans></span></div>
            <input
              type="text"
              name="replacement"
              className="input w-full"
              defaultValue={state.replacement}
              key={state === initialState ? "reset-replacement" : undefined}
              placeholder={t`e.g. GitHub…`}
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={isPending}>
            {isPending ? <span className="loading loading-spinner loading-sm" /> : <Trans>Add</Trans>}
          </button>
        </fieldset>
      </section>
    </form>
  );
}
