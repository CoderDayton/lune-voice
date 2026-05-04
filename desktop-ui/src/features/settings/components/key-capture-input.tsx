import { useEffect, useRef, useState } from "react";
import { cn } from "../../../lib/cn";
import { buildCombo, validateCombo } from "./hotkey-utils";

interface KeyCaptureInputProps {
  value: string;
  onChange: (combo: string) => void;
  className?: string;
  disabled?: boolean;
}

export function KeyCaptureInput({
  value,
  onChange,
  className,
  disabled = false,
}: KeyCaptureInputProps) {
  const [listening, setListening] = useState(false);
  const [pendingCombo, setPendingCombo] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const prevValueRef = useRef(value);

  // Keep prevValueRef up to date when not in listening mode.
  useEffect(() => {
    if (!listening) {
      prevValueRef.current = value;
    }
  }, [value, listening]);

  useEffect(() => {
    if (!listening) return;

    function handleKeyDown(event: KeyboardEvent) {
      event.preventDefault();
      event.stopPropagation();

      if (event.key === "Escape") {
        // Cancel: restore previous value without persisting.
        setPendingCombo(null);
        setValidationError(null);
        setListening(false);
        return;
      }

      const combo = buildCombo(event);
      if (!combo) return;

      const error = validateCombo(combo);
      if (error) {
        setValidationError(error);
        setPendingCombo(combo);
        return;
      }

      setValidationError(null);
      setPendingCombo(null);
      setListening(false);
      onChange(combo);
    }

    window.addEventListener("keydown", handleKeyDown, { capture: true });
    return () => window.removeEventListener("keydown", handleKeyDown, { capture: true });
  }, [listening, onChange]);

  function handleClick() {
    if (disabled) return;
    setPendingCombo(null);
    setValidationError(null);
    setListening(true);
  }

  const displayValue = listening
    ? (pendingCombo ?? "…")
    : value;

  return (
    <div className={cn("flex flex-col items-end gap-1 sm:gap-1.5", className)}>
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled}
        aria-label={listening ? "Press keys to set hotkey" : `Current binding: ${value}. Click to change.`}
        aria-invalid={validationError ? "true" : undefined}
        aria-describedby={validationError ? "key-capture-error" : undefined}
        className={cn(
          "btn btn-outline btn-sm font-mono font-semibold select-none",
          "min-w-28 sm:min-w-36 justify-center",
          listening
            ? "btn-primary animate-pulse"
            : "btn-ghost",
          disabled && "btn-disabled",
        )}
      >
        {listening ? (
          <>
            <span className="h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
            <span>{pendingCombo ? pendingCombo.replace(/\+/g, " + ") : "Press keys…"}</span>
          </>
        ) : (
          <span>{displayValue.replace(/\+/g, " + ")}</span>
        )}
      </button>

      {listening && !validationError && (
        <p className="text-sm text-content-subtle">
          Press your combo — <kbd className="kbd kbd-sm">Esc</kbd> to cancel
        </p>
      )}

      {validationError && (
        <p id="key-capture-error" className="text-sm text-error max-w-52 sm:max-w-64 text-right">
          {validationError}
        </p>
      )}
    </div>
  );
}
