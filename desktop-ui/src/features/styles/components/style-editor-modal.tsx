import { useLingui } from '@lingui/react/macro';
import { Copy } from 'lucide-react';
import type { RefObject } from 'react';

interface StyleEditorModalProps {
  dialogRef: RefObject<HTMLDialogElement | null>;
  mode: 'create' | 'edit' | 'view';
  name: string;
  description: string;
  instructions: string;
  saving: boolean;
  generating: boolean;
  onNameChange: (v: string) => void;
  onDescriptionChange: (v: string) => void;
  onInstructionsChange: (v: string) => void;
  onClose: () => void;
  onSave: () => void;
  onDuplicate: () => void;
}

export function StyleEditorModal({
  dialogRef,
  mode,
  name,
  description,
  instructions,
  saving,
  generating,
  onNameChange,
  onDescriptionChange,
  onInstructionsChange,
  onClose,
  onSave,
  onDuplicate,
}: StyleEditorModalProps) {
  const { t } = useLingui();
  const isViewOnly = mode === 'view';
  const isEdit = mode === 'edit';
  const instrLen = instructions.length;
  const busy = saving || generating;
  const canSave = !!name.trim() && !!instructions.trim() && !busy;

  return (
    <dialog ref={dialogRef} className="modal" onClose={onClose}>
      <div className="modal-box max-w-lg overflow-hidden border border-surface-border p-5 sm:p-6">
        <h3 className="font-semibold text-lg mb-4">
          {isViewOnly ? t`View style` : isEdit ? t`Edit style` : t`Create style`}
        </h3>
        <div className="flex flex-col gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-content-default">{t`Name`}</span>
            <input
              type="text"
              className="input input-bordered w-full"
              maxLength={64}
              placeholder={t`e.g. Meeting notes`}
              value={name}
              readOnly={isViewOnly}
              onChange={(e) => onNameChange(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-content-default">{t`Description`}</span>
            <input
              type="text"
              className="input input-bordered w-full"
              maxLength={256}
              placeholder={t`One-line summary shown on the card`}
              value={description}
              readOnly={isViewOnly}
              onChange={(e) => onDescriptionChange(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-content-default">{t`Instructions`}</span>
              <span
                className={`text-xs tabular-nums ${instrLen > 1900 ? 'text-warning' : 'text-content-dim'}`}
              >
                {instrLen}/2048
              </span>
            </div>
            <textarea
              className="textarea textarea-bordered w-full min-h-32 resize-y leading-relaxed"
              maxLength={2048}
              placeholder={t`Describe how text should be formatted. E.g., "Use bullet points. Keep sentences short. Avoid jargon."`}
              value={instructions}
              readOnly={isViewOnly}
              onChange={(e) => onInstructionsChange(e.target.value)}
            />
          </label>
        </div>
        <div className="modal-action mt-6">
          {isViewOnly ? (
            <>
              <button type="button" className="btn btn-ghost" onClick={onClose}>
                {t`Close`}
              </button>
              <button type="button" className="btn btn-primary gap-2" onClick={onDuplicate}>
                <Copy size={15} /> {t`Duplicate`}
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn btn-ghost" onClick={onClose} disabled={busy}>
                {t`Cancel`}
              </button>
              <button
                type="button"
                className="btn btn-primary"
                disabled={!canSave}
                onClick={onSave}
              >
                {busy && <span className="loading loading-spinner loading-sm" />}
                {generating ? t`Generating examples…` : t`Save`}
              </button>
            </>
          )}
        </div>
      </div>
      {!busy && (
        <form method="dialog" className="modal-backdrop">
          <button type="submit">{t`close`}</button>
        </form>
      )}
    </dialog>
  );
}
