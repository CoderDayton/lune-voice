// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { useCallback, useRef } from 'react';
import { backendClient } from '../../lib/backend/client';
import { AudioDots } from './audio-dots';
import { ArrowUpIcon, PulseIndicator, XIcon } from './pill-icons';
import { usePillDrag } from './use-pill-drag';
import { usePillState } from './use-pill-state';

const PILL_DEBOUNCE_MS = 200;

function fire(fn: () => Promise<unknown>, tag: string) {
  fn().catch((e) => console.error(`[pill] ${tag} failed`, e));
}

export function DictationPill() {
  const { t } = useLingui();
  const { mode, audioLevel, locked } = usePillState();
  const { suppressClickRef, dragHandlers } = usePillDrag(locked);
  const cancelRef = useRef(false);
  const lastFireRef = useRef(0);

  const debouncedFire = useCallback((fn: () => Promise<unknown>, tag: string) => {
    const now = Date.now();
    if (now - lastFireRef.current < PILL_DEBOUNCE_MS) return;
    lastFireRef.current = now;
    fire(fn, tag);
  }, []);

  const isIdle = mode === 'idle';
  const isProcessing = mode === 'processing';

  // Clicking the bar toggles dictation: on when idle, off when listening, in any
  // activation mode. It always calls toggle_dictation, which flips on the
  // backend's own is-active state — so a click stops a session no matter how it
  // started (hotkey or this pill) and never desyncs from a stale pill mode.
  // Only processing is a passive read-out — it can't be interrupted mid-finalise.
  const isButton = !isProcessing;

  const handleSurface = (e: React.MouseEvent<HTMLButtonElement>) => {
    // Drop focus: dictated text typed while this window is focused must not
    // re-trigger the control via Space/Enter.
    e.currentTarget.blur();
    // A drag that ends on the surface must not also toggle dictation.
    if (suppressClickRef.current) return;
    debouncedFire(() => backendClient.toggleDictation(), 'toggle');
  };

  const handleCancel = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.currentTarget.blur();
    if (suppressClickRef.current) return; // ignore the click that ends a drag
    cancelRef.current = true;
    debouncedFire(() => backendClient.cancelDictation(), 'cancel');
    setTimeout(() => {
      cancelRef.current = false;
    }, 300);
  };

  const surfaceLabel = isIdle
    ? t`Start dictation`
    : isProcessing
      ? t`Processing`
      : t`Stop dictation`;

  // Idle has no audio to visualise, so it shows a static "ready" mark rather
  // than a flowing waveform. The waveform only runs while actually listening.
  const content = isProcessing ? (
    <PulseIndicator label={t`Processing`} />
  ) : isIdle ? (
    <span className="pill-idle-mark" aria-hidden="true" />
  ) : (
    <AudioDots level={audioLevel} />
  );

  // `pill-<mode>` drives the accent ring in CSS, so "listening" and
  // "finalising" are readable from the bar's edge, not just its contents.
  const rootClass = `pill-root pill-${mode}${cancelRef.current ? ' pill-cancelled' : ''}${locked ? ' pill-locked' : ''}`;
  // Both end controls exist only while listening: idle has nothing to discard
  // or accept, and processing can't be interrupted mid-finalise.
  const showCaps = !isIdle && !isProcessing;

  return (
    <div className={rootClass} {...dragHandlers}>
      {showCaps && (
        <button
          type="button"
          className="pill-cap pill-cap-accept"
          onClick={handleSurface}
          aria-label={t`Stop and transcribe`}
        >
          <ArrowUpIcon />
        </button>
      )}

      {isButton ? (
        <button
          type="button"
          className="pill-surface"
          onClick={handleSurface}
          aria-label={surfaceLabel}
        >
          {content}
        </button>
      ) : (
        <div className="pill-surface" role="status" aria-live="polite" aria-label={surfaceLabel}>
          {content}
        </div>
      )}

      {showCaps && (
        <button
          type="button"
          className="pill-cap pill-cap-discard"
          onClick={handleCancel}
          aria-label={t`Discard dictation`}
        >
          <XIcon />
        </button>
      )}
    </div>
  );
}
