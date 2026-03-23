import { useRef } from "react";
import { useLingui } from "@lingui/react/macro";

interface ModalCancelProps {
  title: string;
  description: string;
  triggerLabel?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
}

export function ModalCancel({
  title,
  description,
  triggerLabel,
  confirmLabel,
  cancelLabel,
  onConfirm,
}: ModalCancelProps) {
  const { t } = useLingui();
  const dialogRef = useRef<HTMLDialogElement>(null);

  return (
    <>
      <button
        type="button"
        className="btn btn-ghost btn-sm"
        onClick={() => dialogRef.current?.showModal()}
      >
        {triggerLabel ?? t`Remove`}
      </button>

      <dialog ref={dialogRef} className="modal">
        <div className="modal-box">
          <h3 className="text-lg font-bold">{title}</h3>
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
                dialogRef.current?.close();
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
