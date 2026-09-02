import type { MessageDescriptor } from '@lingui/core';
import { Trans, useLingui } from '@lingui/react/macro';
import { useEffect, useRef, useState } from 'react';
import { cn } from '../../../lib/cn';
import { buildCombo, validateCombo } from './hotkey-utils';

/** How long a bare Escape must be held before it aborts the capture. */
const CANCEL_HOLD_MS = 600;

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
  const { t, i18n } = useLingui();
  const [listening, setListening] = useState(false);
  const [pendingCombo, setPendingCombo] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<MessageDescriptor | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const cancelHoldRef = useRef<number | null>(null);

  useEffect(() => {
    if (!listening) return;

    function clearHold() {
      if (cancelHoldRef.current === null) return;
      window.clearTimeout(cancelHoldRef.current);
      cancelHoldRef.current = null;
    }

    function commit(combo: string) {
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

    function handleKeyDown(event: KeyboardEvent) {
      event.preventDefault();
      event.stopPropagation();
      if (event.repeat) return;

      const combo = buildCombo(event);
      if (!combo) return;

      // Escape stays bindable — it is the default for cancelling a recording —
      // so it cannot simply mean "abort capture". A tap binds it; holding it
      // aborts. That hold is the keyboard-only way out, since every key here is
      // swallowed and the other two exits (click the button again, click away)
      // are mouse-only.
      if (combo === 'Escape') {
        setValidationError(null);
        setPendingCombo(combo);
        clearHold();
        cancelHoldRef.current = window.setTimeout(() => {
          cancelHoldRef.current = null;
          setPendingCombo(null);
          setListening(false);
        }, CANCEL_HOLD_MS);
        return;
      }

      commit(combo);
    }

    function handleKeyUp(event: KeyboardEvent) {
      if (event.key !== 'Escape' || cancelHoldRef.current === null) return;
      clearHold();
      commit('Escape');
    }

    window.addEventListener('keydown', handleKeyDown, { capture: true });
    window.addEventListener('keyup', handleKeyUp, { capture: true });
    return () => {
      clearHold();
      window.removeEventListener('keydown', handleKeyDown, { capture: true });
      window.removeEventListener('keyup', handleKeyUp, { capture: true });
    };
  }, [listening, onChange]);

  // Clicking anywhere outside the control abandons the capture — the pointer
  // equivalent of holding Escape.
  useEffect(() => {
    if (!listening) return;

    function handlePointerDown(event: PointerEvent) {
      if (containerRef.current?.contains(event.target as Node)) return;
      setPendingCombo(null);
      setValidationError(null);
      setListening(false);
    }

    window.addEventListener('pointerdown', handlePointerDown, { capture: true });
    return () => window.removeEventListener('pointerdown', handlePointerDown, { capture: true });
  }, [listening]);

  function handleClick() {
    if (disabled) return;
    setPendingCombo(null);
    setValidationError(null);
    setListening((active) => !active);
  }

  const displayValue = listening ? (pendingCombo ?? '…') : value;

  return (
    <div ref={containerRef} className={cn('flex flex-col items-end gap-1 sm:gap-1.5', className)}>
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled}
        aria-label={
          listening
            ? t`Press keys to set hotkey. Hold Escape to cancel, or click again to keep the current one`
            : t`Current binding: ${value}. Click to change.`
        }
        aria-invalid={validationError ? 'true' : undefined}
        aria-describedby={validationError ? 'key-capture-error' : undefined}
        className={cn(
          'inline-flex h-10 min-w-28 select-none items-center justify-center gap-2 rounded-field border px-4 font-mono text-sm font-semibold tracking-[0.02em] transition-[background,border-color,box-shadow,color,transform]',
          'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary',
          'sm:min-w-36',
          listening
            ? 'animate-pulse border-primary/55 bg-primary/18 text-content-strong shadow-glow'
            : 'border-surface-border-strong bg-surface-elevated text-content shadow-soft hover:border-surface-border-strong hover:bg-surface-hover active:translate-y-px',
          disabled && 'cursor-not-allowed opacity-50',
        )}
      >
        {listening ? (
          <>
            <span className="h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
            <span>{pendingCombo ? pendingCombo.replace(/\+/g, ' + ') : t`Press keys…`}</span>
          </>
        ) : (
          <span>{displayValue.replace(/\+/g, ' + ')}</span>
        )}
      </button>

      {listening && !validationError && (
        <p className="text-sm text-content-subtle">
          <Trans>Press your combo — hold Esc or click away to cancel</Trans>
        </p>
      )}

      {validationError && (
        <p id="key-capture-error" className="text-sm text-error max-w-52 sm:max-w-64 text-right">
          {i18n._(validationError)}
        </p>
      )}
    </div>
  );
}
