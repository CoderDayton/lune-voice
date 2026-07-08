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
  // Tracking mode: null while begin_pill_drag is resolving; true = native
  // cursor-pinned ticks (X11/XWayland — Rust re-derives the position from the
  // live pointer each tick, so nothing can drift); false = incremental
  // movementX/Y fallback (layer-shell Wayland exposes no global pointer).
  native: boolean | null;
  begin: Promise<boolean> | null;
}

/**
 * Makes the pill overlay draggable.
 *
 * Two tracking modes, negotiated per drag via `begin_pill_drag`:
 *
 * - **Native (X11/XWayland):** Rust anchors the drag to the pointer's root
 *   position and, on every `drag_pill_tick`, places the window at that anchor
 *   offset from the *current* pointer. Ticks are absolute, so IPC latency and
 *   coalesced events cost at most transient lag — never cumulative drift. The
 *   old delta-only path integrated integer-truncated `movementX/Y` through
 *   async IPC, which visibly wandered off the cursor over a long drag.
 * - **Fallback (Wayland layer-shell):** the compositor gives no global pointer
 *   position, so per-event `movementX/Y` deltas are accumulated and flushed as
 *   `nudge_pill` margin nudges, one per animation frame, as before.
 *
 * On pointer-up after a real drag the position is persisted
 * (`savePillPosition`) and the interaction is flagged via `suppressClickRef`
 * so the surface's click handler doesn't also toggle dictation.
 */
export function usePillDrag() {
  const stateRef = useRef<DragState | null>(null);
  const pendingRef = useRef({ dx: 0, dy: 0 });
  const frameRef = useRef<number | null>(null);
  const tickBusyRef = useRef(false);
  const suppressClickRef = useRef(false);

  const flushDeltas = () => {
    const { dx, dy } = pendingRef.current;
    pendingRef.current = { dx: 0, dy: 0 };
    if (dx === 0 && dy === 0) return Promise.resolve();
    return backendClient.nudgePill(dx, dy);
  };

  const onFrame = () => {
    frameRef.current = null;
    const st = stateRef.current;
    if (!st?.moved) return;
    if (st.native === true) {
      // Rust reads the live pointer itself; this call is just the heartbeat
      // that paces it. Serialize — a queued backlog adds lag without adding
      // information, since every tick is absolute.
      if (tickBusyRef.current) {
        schedule();
        return;
      }
      tickBusyRef.current = true;
      backendClient
        .dragPillTick()
        .catch((e) => console.error('[pill] drag tick failed', e))
        .finally(() => {
          tickBusyRef.current = false;
        });
    } else if (st.native === false) {
      flushDeltas().catch((e) => console.error('[pill] nudge failed', e));
    }
    // native === null: begin_pill_drag is still resolving — its handler
    // reschedules once the mode is known.
  };

  const schedule = () => {
    frameRef.current ??= requestAnimationFrame(onFrame);
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
      native: null,
      begin: null,
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
      // Seed the fallback with the pre-threshold movement, and hand the same
      // displacement to begin_pill_drag as catch-up for the native anchor.
      pendingRef.current.dx += st.accDx;
      pendingRef.current.dy += st.accDy;
      st.begin = backendClient
        .beginPillDrag(st.accDx, st.accDy)
        .catch((err) => {
          console.error('[pill] begin drag failed', err);
          return false;
        })
        .then((ok) => {
          if (stateRef.current === st) {
            st.native = ok;
            // Native anchor already includes the catch-up; drop the seed so a
            // later fallback flush can't double-apply it.
            if (ok) pendingRef.current = { dx: 0, dy: 0 };
            schedule();
          }
          return ok;
        });
      return;
    }
    if (st.native === true) {
      schedule();
      return;
    }
    pendingRef.current.dx += dx;
    pendingRef.current.dy += dy;
    if (st.native === false) schedule();
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
    // Settle exactly, then persist the resting spot: native mode ends with a
    // final anchored tick in Rust; the fallback flushes any remaining delta.
    (st.begin ?? Promise.resolve(false))
      .then((native) => (native ? backendClient.endPillDrag() : flushDeltas()))
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
