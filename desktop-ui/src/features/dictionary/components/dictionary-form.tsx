import { Trans, useLingui } from '@lingui/react/macro';
import { Plus } from 'lucide-react';
import { useActionState, useRef, useState } from 'react';
import { addDictionaryEntry } from '../../../store/workspace-store';
import { PriorityPicker } from './priority-picker';

interface FormState {
  phrase: string;
  replacement: string;
  error: string | null;
}

const initialState: FormState = { phrase: '', replacement: '', error: null };

export function DictionaryForm() {
  const { t } = useLingui();
  const resetCounter = useRef(0);
  const [priority, setPriority] = useState(5);

  const [state, submitAction, isPending] = useActionState(
    async (_prev: FormState, formData: FormData) => {
      const phrase = (formData.get('phrase') as string) ?? '';
      const replacement = (formData.get('replacement') as string) ?? '';
      const p = Number(formData.get('priority') ?? 5);
      const added = await addDictionaryEntry(phrase, replacement, p);
      if (added) {
        resetCounter.current++;
        return initialState;
      }
      return {
        phrase,
        replacement,
        error: 'Failed to add entry. Please try again.',
      };
    },
    initialState,
  );

  return (
    <form action={submitAction}>
      <input type="hidden" name="priority" value={priority} />
      <section className="panel-surface p-5 sm:p-6">
        <h3 className="text-xl font-semibold tracking-tight mb-4">
          <Trans>Add correction rule</Trans>
        </h3>
        <fieldset className="flex flex-col gap-4" disabled={isPending}>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <label className="flex flex-1 flex-col gap-1.5">
              <span className="text-sm font-medium">
                <Trans>Spoken phrase</Trans>
              </span>
              <input
                type="text"
                name="phrase"
                className="input w-full"
                defaultValue={state.phrase}
                key={`phrase-${resetCounter.current}`}
                placeholder={t`e.g. github…`}
                required
                aria-required="true"
              />
            </label>
            <span className="hidden text-lg text-content-dim sm:block sm:pb-2.5" aria-hidden="true">
              →
            </span>
            <label className="flex flex-1 flex-col gap-1.5">
              <span className="text-sm font-medium">
                <Trans>Replacement</Trans>
              </span>
              <input
                type="text"
                name="replacement"
                className="input w-full"
                defaultValue={state.replacement}
                key={`replacement-${resetCounter.current}`}
                placeholder={t`e.g. GitHub…`}
                required
                aria-required="true"
              />
            </label>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium">
                <Trans>Priority</Trans>
              </span>
              <PriorityPicker value={priority} onChange={setPriority} size="sm" />
            </div>
            <button
              type="submit"
              className="btn btn-primary btn-sm w-full sm:w-auto sm:min-w-32"
              disabled={isPending}
            >
              {isPending ? (
                <span className="loading loading-spinner loading-sm" />
              ) : (
                <>
                  <Plus size={14} strokeWidth={2.5} />
                  <Trans>Add</Trans>
                </>
              )}
            </button>
          </div>
        </fieldset>
        {state.error && (
          <p role="alert" className="mt-3 text-sm text-error">
            {state.error}
          </p>
        )}
      </section>
    </form>
  );
}
