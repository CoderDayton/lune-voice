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
 * Makes the pill overlay draggable. The pill window is a Wayland layer-shell
 * surface that the compositor can't move, so pointer movement is translated
 * into incremental margin nudges on the Rust side (`backendClient.nudgePill`),
 * coalesced to one call per animation frame. On pointer-up after a real drag
 * the position is persisted (`savePillPosition`) and the interaction is flagged
 * via `suppressClickRef` so the surface's click handler doesn't also toggle
 * dictation.
 *
 * Per-event `movementX/Y` deltas are used rather than absolute coordinates: the
 * window follows the cursor during a drag (so absolute client coordinates
 * barely change), but the deltas keep accumulating correctly.
 */
export function usePillDrag() {
  const stateRef = useRef<DragState | null>(null);
  const pendingRef = useRef({ dx: 0, dy: 0 });
  const frameRef = useRef<number | null>(null);
  const suppressClickRef = useRef(false);

  const flush = () => {
    frameRef.current = null;
    const { dx, dy } = pendingRef.current;
    pendingRef.current = { dx: 0, dy: 0 };
    if (dx !== 0 || dy !== 0) {
      backendClient.nudgePill(dx, dy).catch((e) => console.error('[pill] nudge failed', e));
    }
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return; // primary button / touch only
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
      // leaves the small pill window.
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      st.captured = true;
      // Catch the pill up to the accumulated pre-threshold movement so it
      // doesn't lag behind the cursor by the threshold amount.
      pendingRef.current.dx += st.accDx;
      pendingRef.current.dy += st.accDy;
      frameRef.current ??= requestAnimationFrame(flush);
      return;
    }
    pendingRef.current.dx += dx;
    pendingRef.current.dy += dy;
    frameRef.current ??= requestAnimationFrame(flush);
  };

  const endDrag = (e: React.PointerEvent) => {
    const st = stateRef.current;
    if (!st || e.pointerId !== st.pointerId) return;
    if (st.captured) {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    }
    stateRef.current = null;
    // Flush any pending delta immediately before persisting.
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    flush();
    if (st.moved) {
      suppressClickRef.current = true;
      backendClient
        .savePillPosition()
        .catch((err) => console.error('[pill] save position failed', err));
    }
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
