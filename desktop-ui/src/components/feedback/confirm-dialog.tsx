import { useState } from "react";
import { useLingui } from "@lingui/react/macro";

interface ConfirmDialogProps {
  title: string;
  description: string;
  triggerLabel?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
}

export function ConfirmDialog({
  title,
  description,
  triggerLabel,
  confirmLabel,
  cancelLabel,
  onConfirm,
}: ConfirmDialogProps) {
  const { t } = useLingui();
  const [dialog, setDialog] = useState<HTMLDialogElement | null>(null);

  return (
    <>
      <button
        type="button"
        className="btn btn-ghost btn-sm"
        aria-haspopup="dialog"
        onClick={() => dialog?.showModal()}
      >
        {triggerLabel ?? t`Remove`}
      </button>

      <dialog
        ref={(el) => {
          setDialog(el);
          return () => setDialog(null);
        }}
        className="modal"
        aria-labelledby="confirm-dialog-title"
      >
        <div className="modal-box">
          <h3 id="confirm-dialog-title" className="text-lg font-bold">{title}</h3>
          <p className="py-4 text-sm opacity-70">{description}</p>
          <div className="modal-action">
            <form method="dialog">
              <button type="submit" className="btn btn-ghost">{cancelLabel ?? t`Cancel`}</button>
            </form>
            <button
              type="button"
              className="btn btn-error"
              onClick={() => {
                onConfirm();
                dialog?.close();
              }}
            >
              {confirmLabel ?? t`Delete`}
            </button>
          </div>
        </div>
        <form method="dialog" className="modal-backdrop"><button type="submit">close</button></form>
      </dialog>
    </>
  );
}
