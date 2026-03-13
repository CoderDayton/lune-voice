import { createContext, type ReactNode, useContext, useEffect, useId } from "react";
import { createPortal } from "react-dom";
import { cn } from "../../lib/cn";

type ModalSize = "sm" | "md" | "lg";

const sizeClasses: Record<ModalSize, string> = {
  sm: "max-w-sm",
  md: "max-w-lg",
  lg: "max-w-3xl",
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
      className="modal modal-open"
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
        className={cn("modal-box mx-4 sm:mx-auto", sizeClasses[size])}
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
    <div className="flex items-start justify-between gap-3 sm:gap-4 border-b border-base-300/40 pb-4">
      <div className="min-w-0">
        <h2 id={titleId} className="text-lg font-semibold">
          {title}
        </h2>
        {description ? <p className="body-copy-soft mt-1">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function ModalBody({ children, className }: ModalBodyProps) {
  return <div className={cn("py-4", className)}>{children}</div>;
}

export function ModalFooter({ children, className }: ModalFooterProps) {
  return <div className={cn("modal-action border-t border-base-300/40 pt-4", className)}>{children}</div>;
}
