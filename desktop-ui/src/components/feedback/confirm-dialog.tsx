import { useLingui } from '@lingui/react/macro';
import { type ReactNode, useId, useState } from 'react';

interface ConfirmDialogProps {
  title: string;
  description: string;
  triggerLabel?: string;
  /** Render the trigger as an icon button; `triggerLabel` becomes its
   *  accessible name instead of visible text. */
  triggerIcon?: ReactNode;
  triggerClassName?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
}

export function ConfirmDialog({
  title,
  description,
  triggerLabel,
  triggerIcon,
  triggerClassName,
  confirmLabel,
  cancelLabel,
  onConfirm,
}: ConfirmDialogProps) {
  const { t } = useLingui();
  const titleId = useId();
  const [dialog, setDialog] = useState<HTMLDialogElement | null>(null);

  return (
    <>
      <button
        type="button"
        className={triggerClassName ?? 'btn btn-soft btn-sm'}
        aria-haspopup="dialog"
        aria-label={triggerIcon ? (triggerLabel ?? t`Remove`) : undefined}
        onClick={() => dialog?.showModal()}
      >
        {triggerIcon ?? triggerLabel ?? t`Remove`}
      </button>

      <dialog
        ref={(el) => {
          setDialog(el);
          return () => setDialog(null);
        }}
        className="modal"
        aria-labelledby={titleId}
      >
        <div className="modal-box border border-surface-border">
          <h3 id={titleId} className="text-xl font-bold">
            {title}
          </h3>
          <p className="py-4 text-sm text-content-muted">{description}</p>
          <div className="modal-action">
            <form method="dialog">
              <button type="submit" className="btn btn-soft">
                {cancelLabel ?? t`Cancel`}
              </button>
            </form>
            <button
              type="button"
              className="btn btn-soft btn-error"
              onClick={() => {
                onConfirm();
                dialog?.close();
              }}
            >
              {confirmLabel ?? t`Delete`}
            </button>
          </div>
        </div>
        <form method="dialog" className="modal-backdrop">
          <button type="submit">close</button>
        </form>
      </dialog>
    </>
  );
}
