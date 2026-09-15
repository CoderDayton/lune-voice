// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useRef } from 'react';
import { backendClient } from '../../lib/backend/client';

// Pointer movement (px) beyond which an interaction counts as a drag rather
// than a click. Below this, pointer-up still toggles dictation as usual.
const DRAG_THRESHOLD = 3;

interface DragState {
  pointerId: number;
  moved: boolean;
  captured: boolean;
  // Displacement accumulated since pointerdown, while still below the drag
  // threshold. `movementX/Y` is the delta since the *previous* event, so a
  // slow drag can produce many sub-threshold events in a row — the threshold
  // must be tested against this running total, not each event in isolation.
  accDx: number;
  accDy: number;
}

/**
 * Makes the webview pill overlay draggable (Windows/macOS — the Linux pill is
 * a native GTK window that handles its own pointer events).
 *
 * `movementX/Y` deltas are accumulated and flushed as `nudge_pill` calls, one
 * per animation frame, so a burst of pointer events costs one IPC round trip
 * rather than dozens.
 *
 * On pointer-up after a real drag the position is persisted
 * (`savePillPosition`) and the interaction is flagged via `suppressClickRef`
 * so the surface's click handler doesn't also toggle dictation.
 *
 * `locked` (the "Lock position" setting) disables dragging entirely — pointer
 * interactions stay plain clicks. The backend refuses nudges too, so a stale
 * flag can't move the pill.
 */
export function usePillDrag(locked = false) {
  const stateRef = useRef<DragState | null>(null);
  const pendingRef = useRef({ dx: 0, dy: 0 });
  const frameRef = useRef<number | null>(null);
  const suppressClickRef = useRef(false);

  const flushDeltas = () => {
    const { dx, dy } = pendingRef.current;
    pendingRef.current = { dx: 0, dy: 0 };
    if (dx === 0 && dy === 0) return Promise.resolve();
    return backendClient.nudgePill(dx, dy);
  };

  const schedule = () => {
    frameRef.current ??= requestAnimationFrame(() => {
      frameRef.current = null;
      if (!stateRef.current?.moved) return;
      flushDeltas().catch((e) => console.error('[pill] nudge failed', e));
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return; // primary button / touch only
    if (locked) return; // position locked — leave the interaction a plain click
    suppressClickRef.current = false;
    // Don't capture the pointer yet: capturing on down redirects the pointer to
    // this element, so the subsequent `click` never reaches the inner surface
    // button and dictation never toggles. Capture is taken lazily once a real
    // drag begins (onPointerMove), so a plain click still clicks.
    stateRef.current = {
      pointerId: e.pointerId,
      moved: false,
      captured: false,
      accDx: 0,
      accDy: 0,
    };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const st = stateRef.current;
    if (!st || e.pointerId !== st.pointerId) return;
    const dx = e.movementX;
    const dy = e.movementY;
    if (dx === 0 && dy === 0) return;
    if (!st.moved) {
      // Below the threshold it's still a potential click — don't move the pill.
      // Test cumulative displacement since pointerdown, not this event alone.
      st.accDx += dx;
      st.accDy += dy;
      if (Math.abs(st.accDx) + Math.abs(st.accDy) < DRAG_THRESHOLD) return;
      st.moved = true;
      // Now it's a drag: capture so movement keeps arriving even if the cursor
      // leaves the small pill window, and seed the flush with the pre-threshold
      // movement so the pill catches up to the original grab point.
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      st.captured = true;
      pendingRef.current.dx += st.accDx;
      pendingRef.current.dy += st.accDy;
    } else {
      pendingRef.current.dx += dx;
      pendingRef.current.dy += dy;
    }
    schedule();
  };

  const endDrag = (e: React.PointerEvent) => {
    const st = stateRef.current;
    if (!st || e.pointerId !== st.pointerId) return;
    if (st.captured) {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    }
    stateRef.current = null;
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    if (!st.moved) return;
    suppressClickRef.current = true;
    // Settle exactly, then persist the resting spot.
    flushDeltas()
      .then(() => backendClient.savePillPosition())
      .catch((err) => console.error('[pill] end drag failed', err));
  };

  // Cancel any pending frame on unmount.
  useEffect(
    () => () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    },
    [],
  );

  return {
    suppressClickRef,
    dragHandlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
    },
  };
}
