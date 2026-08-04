import { describe, expect, test } from "bun:test";
import type { Clock } from "../../src/features/dictation/mode-gate";
import { ModeGate } from "../../src/features/dictation/mode-gate";
import type { PillMode } from "../../src/features/dictation/use-pill-state";

/** Deterministic clock: time only moves when advance() is called. */
function manualClock() {
  let now = 0;
  let nextId = 1;
  const pending = new Map<number, { at: number; cb: () => void }>();
  const clock: Clock = {
    now: () => now,
    setTimeout(cb, ms) {
      const id = nextId++;
      pending.set(id, { at: now + ms, cb });
      return id;
    },
    clearTimeout(id) {
      pending.delete(id);
    },
  };
  return {
    clock,
    // Fires timers in chronological order, with `now` set to each one's
    // deadline as it runs — a timer that fires must not observe a clock that
    // has already run past it, or every callback looks arbitrarily late.
    advance(ms: number) {
      const target = now + ms;
      for (;;) {
        const next = [...pending.entries()]
          .filter(([, t]) => t.at <= target)
          .sort(([, a], [, b]) => a.at - b.at)[0];
        if (!next) break;
        const [id, { at, cb }] = next;
        pending.delete(id);
        now = at;
        cb();
      }
      now = target;
    },
  };
}

function gate() {
  const { clock, advance } = manualClock();
  const seen: PillMode[] = [];
  return { g: new ModeGate((m) => seen.push(m), clock), seen, advance };
}

describe("ModeGate", () => {
  test("listening shows immediately", () => {
    const { g, seen } = gate();
    g.push("active");
    expect(seen).toEqual(["active"]);
  });

  /** The reported bug: toggle on, say nothing, toggle off. The empty buffer
   *  decodes instantly, so `processing` arrives and leaves within a frame and
   *  used to render as a flash of the accent hue. */
  test("a processing state too brief to read is never drawn", () => {
    const { g, seen, advance } = gate();
    g.push("active");
    g.push("processing");
    advance(20);
    g.push("idle");
    advance(1000);
    expect(seen).toEqual(["active", "idle"]);
  });

  test("a processing state that lasts is drawn, after the debut delay", () => {
    const { g, seen, advance } = gate();
    g.push("active");
    g.push("processing");
    advance(140);
    expect(seen).toEqual(["active"]);
    advance(20);
    expect(seen).toEqual(["active", "processing"]);
  });

  /** Without the dwell, a decode that finishes just past the debut delay would
   *  produce the same flash the debut delay exists to prevent. */
  test("once drawn, processing stays long enough to see", () => {
    const { g, seen, advance } = gate();
    g.push("active");
    g.push("processing");
    advance(150);
    expect(seen).toEqual(["active", "processing"]);
    g.push("idle");
    advance(200);
    expect(seen).toEqual(["active", "processing"]);
    advance(60);
    expect(seen).toEqual(["active", "processing", "idle"]);
  });

  test("a long processing state ends as soon as it is withdrawn", () => {
    const { g, seen, advance } = gate();
    g.push("processing");
    advance(2000);
    g.push("idle");
    expect(seen).toEqual(["processing", "idle"]);
  });

  test("re-pushing the rendered mode is a no-op", () => {
    const { g, seen, advance } = gate();
    g.push("active");
    g.push("active");
    advance(1000);
    expect(seen).toEqual(["active"]);
  });
});
