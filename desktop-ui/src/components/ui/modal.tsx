import { createContext, type ReactNode, useContext, useEffect, useId } from "react";
import { createPortal } from "react-dom";
import { cn } from "../../lib/cn";

type ModalSize = "sm" | "md" | "lg";

const sizeClasses: Record<ModalSize, string> = {
  sm: "modal-panel-sm",
  md: "modal-panel-md",
  lg: "modal-panel-lg",
};

const ModalTitleIdContext = createContext<string | undefined>(undefined);

interface ModalProps {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  size?: ModalSize;
  closeOnBackdrop?: boolean;
  closeOnEscape?: boolean;
  ariaLabel?: string;
}

interface ModalHeaderProps {
  title: string;
  description?: string;
  action?: ReactNode;
}

interface ModalBodyProps {
  children: ReactNode;
  className?: string;
}

interface ModalFooterProps {
  children: ReactNode;
  className?: string;
}

export function Modal({
  open,
  onClose,
  children,
  size = "md",
  closeOnBackdrop = true,
  closeOnEscape = true,
  ariaLabel,
}: ModalProps) {
  const titleId = useId();

  useEffect(() => {
    if (!open || !closeOnEscape) {
      return;
    }

    function handleKeydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [closeOnEscape, onClose, open]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div
      className="modal-overlay"
      onMouseDown={(event) => {
        if (closeOnBackdrop && event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        aria-labelledby={ariaLabel ? undefined : titleId}
        className={cn("modal-panel", sizeClasses[size])}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <ModalTitleIdContext.Provider value={titleId}>{children}</ModalTitleIdContext.Provider>
      </div>
    </div>,
    document.body,
  );
}

export function ModalHeader({ title, description, action }: ModalHeaderProps) {
  const titleId = useContext(ModalTitleIdContext);

  return (
    <div className="modal-header">
      <div className="modal-copy">
        <h2 id={titleId} className="modal-title">
          {title}
        </h2>
        {description ? <p className="modal-description">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function ModalBody({ children, className }: ModalBodyProps) {
  return <div className={cn("modal-body", className)}>{children}</div>;
}

export function ModalFooter({ children, className }: ModalFooterProps) {
  return <div className={cn("modal-footer", className)}>{children}</div>;
}
