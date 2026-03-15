import { useState } from "react";
import { Button } from "./button";
import { Modal, ModalFooter, ModalHeader } from "./modal";

interface ModalCancelProps {
  title: string;
  description: string;
  triggerLabel?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  triggerClassName?: string;
}

export function ModalCancel({
  title,
  description,
  triggerLabel = "Remove",
  confirmLabel = "Delete",
  cancelLabel = "Cancel",
  onConfirm,
  triggerClassName,
}: ModalCancelProps) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <Button type="button" variant="table" onClick={() => setOpen(true)} className={triggerClassName}>
        {triggerLabel}
      </Button>

      <Modal open={open} onClose={() => setOpen(false)} size="sm">
        <ModalHeader title={title} description={description} />
        <ModalFooter>
          <Button type="button" variant="secondary" onClick={() => setOpen(false)}>
            {cancelLabel}
          </Button>
          <Button
            variant="danger"
            onClick={() => {
              onConfirm();
              setOpen(false);
            }}
          >
            {confirmLabel}
          </Button>
        </ModalFooter>
      </Modal>
    </>
  );
}
