import { useActionState } from "react";
import { Check } from "lucide-react";
import { addDictionaryEntry } from "../../../store/workspace-store";
import { Trans, useLingui } from "@lingui/react/macro";

interface FormState {
  phrase: string;
  replacement: string;
  error: string | null;
  success: boolean;
}

const initialState: FormState = { phrase: "", replacement: "", error: null, success: false };

export function DictionaryForm() {
  const { t } = useLingui();

  const [state, submitAction, isPending] = useActionState(
    async (_prev: FormState, formData: FormData) => {
      const phrase = (formData.get("phrase") as string) ?? "";
      const replacement = (formData.get("replacement") as string) ?? "";
      const added = await addDictionaryEntry(phrase, replacement);
      if (added) return { ...initialState, success: true };
      return { phrase, replacement, error: "Failed to add entry. Please try again.", success: false };
    },
    initialState,
  );

  return (
    <form action={submitAction}>
      <section className="card border border-base-content/[0.06] bg-base-200/40 rounded-box p-5 sm:p-6">
        <h3 className="text-lg font-semibold tracking-tight mb-4"><Trans>Add correction rule</Trans></h3>
        <fieldset className="flex flex-col sm:flex-row gap-3 items-end" disabled={isPending}>
          <label className="form-control flex-1">
            <div className="label"><span className="label-text"><Trans>Spoken phrase</Trans></span></div>
            <input
              type="text"
              name="phrase"
              className="input w-full"
              defaultValue={state.phrase}
              key={state.success ? `reset-phrase-${Date.now()}` : undefined}
              placeholder={t`e.g. github…`}
              required
              aria-required="true"
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
              key={state.success ? `reset-replacement-${Date.now()}` : undefined}
              placeholder={t`e.g. GitHub…`}
              required
              aria-required="true"
            />
          </label>
          <button type="submit" className="btn btn-primary w-full sm:w-auto" disabled={isPending}>
            {isPending ? (
              <span className="loading loading-spinner loading-sm" />
            ) : state.success ? (
              <><Check size={16} strokeWidth={2.5} /> <Trans>Added</Trans></>
            ) : (
              <Trans>Add</Trans>
            )}
          </button>
        </fieldset>
        {state.error && (
          <p role="alert" className="mt-3 text-sm text-error">{state.error}</p>
        )}
      </section>
    </form>
  );
}
