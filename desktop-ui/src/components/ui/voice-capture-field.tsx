import { Trans, useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Mic } from 'lucide-react';
import { type ReactNode, type Ref, useEffect, useRef, useState } from 'react';
import {
  cancelDictation,
  startTriggerCapture,
  stopDictation,
  workspaceStore,
} from '../../store/workspace-store';

interface VoiceCaptureFieldProps {
  id: string;
  value: string;
  onValueChange: (value: string) => void;
  /** Idle placeholder; the live capture states supply their own. */
  placeholder: string;
  /** Idle hint line under the field; live capture states replace it. When
   *  omitted, the line only appears while capturing (compact layouts). */
  hint?: ReactNode;
  /** Lowercase captured text (vocabulary phrases are stored lowercase). */
  lowercase?: boolean;
  inputRef?: Ref<HTMLInputElement>;
  required?: boolean;
}

/**
 * Text input with a built-in voice recorder. The mic button runs a
 * capture-only session: speech is transcribed like dictation but nothing is
 * typed into any app, and the field receives the *raw* ASR text — byte-for-
 * byte what the matchers compare against at runtime — so a spoken phrase is
 * stored exactly the way Lune Voice hears it. Used by the snippet trigger
 * composer and the vocabulary "you say" field.
 *
 * Only one capture session exists app-wide; each field instance consumes a
 * result only if it started that capture, so two fields never race.
 */
export function VoiceCaptureField({
  id,
  value,
  onValueChange,
  placeholder,
  hint,
  lowercase,
  inputRef,
  required,
}: VoiceCaptureFieldProps) {
  const { t } = useLingui();
  const captureSessionId = useStore(workspaceStore, (s) => s.runtime.captureSessionId);
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const audioLevel = useStore(workspaceStore, (s) => s.runtime.audioLevel);
  const captureText = useStore(workspaceStore, (s) => s.runtime.captureText);
  const captureSeq = useStore(workspaceStore, (s) => s.runtime.captureSeq);

  // Whether *this* field started the active capture (vs another field/page).
  const owns = useRef(false);
  const capturing = captureSessionId !== null && owns.current;
  const capturePhase = !capturing
    ? 'idle'
    : serviceStatus === 'processing'
      ? 'processing'
      : 'listening';
  // The mic is taken by real dictation or another field's capture.
  const micBusy = !capturing && (serviceStatus === 'listening' || serviceStatus === 'processing');

  const [flash, setFlash] = useState<'hit' | 'empty' | null>(null);
  const lastSeq = useRef(captureSeq);

  useEffect(() => {
    if (captureSeq === lastSeq.current) return;
    lastSeq.current = captureSeq;
    if (!owns.current) return; // another field's capture — not ours to consume
    owns.current = false;
    // Raw whisper output often carries a sentence capital and terminal
    // period; both matchers ignore punctuation, so strip the trailing run
    // for a clean stored phrase. Everything else stays exactly as heard.
    let heard = (captureText ?? '').trim().replace(/[.,!?;:]+$/u, '');
    if (lowercase) heard = heard.toLowerCase();
    if (heard) {
      onValueChange(heard);
      setFlash('hit');
    } else {
      setFlash('empty');
    }
    const timer = setTimeout(() => setFlash(null), 1400);
    return () => clearTimeout(timer);
  }, [captureSeq, captureText, onValueChange, lowercase]);

  // A session that ended without a result (cancelled / errored) must drop
  // ownership, or this field would consume a *later* capture started by
  // another field. After a successful result this is a no-op (already false).
  useEffect(() => {
    if (captureSessionId === null) owns.current = false;
  }, [captureSessionId]);

  function toggleCapture() {
    setFlash(null);
    if (capturing) {
      void stopDictation();
    } else {
      owns.current = true;
      void startTriggerCapture().then((sessionId) => {
        if (!sessionId) owns.current = false; // failed to start
      });
    }
  }

  return (
    <>
      <div
        className={`capture-field ${capturing ? 'capture-field-live' : ''} ${
          flash === 'hit' ? 'capture-field-hit' : ''
        }`}
      >
        <input
          id={id}
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape' && capturing) {
              e.preventDefault();
              void cancelDictation();
            }
          }}
          readOnly={capturing}
          className="input input-lg w-full pr-14 text-base"
          placeholder={
            capturePhase === 'listening'
              ? t`Listening — say the phrase…`
              : capturePhase === 'processing'
                ? t`Transcribing…`
                : placeholder
          }
          autoComplete="off"
          autoCapitalize="off"
          spellCheck={false}
          required={required}
          aria-required={required ? 'true' : undefined}
        />
        <button
          type="button"
          onClick={toggleCapture}
          disabled={micBusy}
          className={`capture-btn ${capturePhase === 'listening' ? 'capture-btn-listening' : ''}`}
          aria-pressed={capturing}
          aria-label={capturing ? t`Stop capturing` : t`Capture by voice`}
          title={
            micBusy
              ? t`The microphone is in use`
              : capturing
                ? t`Stop capturing`
                : t`Capture by voice`
          }
        >
          {capturePhase === 'listening' ? (
            <CaptureBars level={audioLevel} />
          ) : capturePhase === 'processing' ? (
            <span className="loading loading-spinner loading-xs" />
          ) : (
            <Mic size={16} strokeWidth={2} />
          )}
        </button>
      </div>
      <CaptureStatusLine phase={capturePhase} flash={flash} hint={hint} />
    </>
  );
}

/**
 * Status line under the field: live capture states take priority, then the
 * caller's idle hint. Renders nothing when idle with no hint, so compact
 * layouts (vocabulary composer) stay tight until a capture starts.
 */
function CaptureStatusLine({
  phase,
  flash,
  hint,
}: {
  phase: 'idle' | 'listening' | 'processing';
  flash: 'hit' | 'empty' | null;
  hint?: ReactNode;
}) {
  const content =
    phase === 'listening' ? (
      <Trans>Listening — say the phrase, then click again to stop.</Trans>
    ) : phase === 'processing' ? (
      <Trans>Transcribing what you said…</Trans>
    ) : flash === 'empty' ? (
      <Trans>Didn't catch anything — try again, a little closer to the mic.</Trans>
    ) : (
      (hint ?? null)
    );
  if (!content) return null;
  return (
    <p
      aria-live="polite"
      className={`flex items-start gap-1.5 text-sm ${
        flash === 'empty' ? 'text-warning' : 'text-content-subtle'
      }`}
    >
      <Mic size={14} strokeWidth={2} className="mt-0.5 shrink-0" aria-hidden="true" />
      {content}
    </p>
  );
}

/**
 * Mini level meter inside the live capture button. Bar heights are driven by
 * the session's real audio level; per-bar weights fan the response so it
 * reads as a waveform rather than four synchronized bars.
 */
function CaptureBars({ level }: { level: number }) {
  const drive = Math.min(1, level * 1.6);
  return (
    <span className="capture-bars" aria-hidden="true">
      {[0.55, 1, 0.8, 0.4].map((weight) => (
        <span
          key={weight}
          className="capture-bar"
          style={{ transform: `scaleY(${(0.25 + drive * weight * 0.75).toFixed(3)})` }}
        />
      ))}
    </span>
  );
}
