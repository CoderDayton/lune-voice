import { describe, expect, test } from "bun:test";
import { DOT_COUNT, dotStyle } from "../../src/features/dictation/dot-style";
import {
  type FrameScheduler,
  LevelSmoother,
} from "../../src/features/dictation/level-smoother";

/** Deterministic scheduler: frames fire only when flush() is called. */
function manualScheduler() {
  let nextId = 1;
  const pending = new Map<number, () => void>();
  const scheduler: FrameScheduler = {
    request(cb) {
      const id = nextId++;
      pending.set(id, cb);
      return id;
    },
    cancel(id) {
      pending.delete(id);
    },
  };
  return {
    scheduler,
    flush() {
      const cbs = [...pending.values()];
      pending.clear();
      for (const cb of cbs) cb();
    },
    pendingCount: () => pending.size,
  };
}

describe("LevelSmoother", () => {
  test("coalesces an event flood into a single commit per frame", () => {
    const frames = manualScheduler();
    const commits: number[] = [];
    const smoother = new LevelSmoother((v) => commits.push(v), frames.scheduler);

    // 100 events land before the next frame — e.g. a backend burst. Without
    // coalescing this caused one React render + style write per event.
    for (let i = 0; i < 100; i++) smoother.push(0.9);

    expect(commits.length).toBe(0); // nothing until the frame fires
    expect(frames.pendingCount()).toBe(1); // exactly one frame scheduled

    frames.flush();
    expect(commits.length).toBe(1); // one commit for the whole flood
    expect(commits[0]).toBeGreaterThan(0.85); // converged near the input

    frames.flush();
    expect(commits.length).toBe(1); // no trailing commits without new events
  });

  test("commits the latest smoothed value once per frame at steady cadence", () => {
    const frames = manualScheduler();
    const commits: number[] = [];
    const smoother = new LevelSmoother((v) => commits.push(v), frames.scheduler);

    for (let tick = 0; tick < 10; tick++) {
      smoother.push(0.5);
      frames.flush();
    }
    expect(commits.length).toBe(10);
  });

  test("attack is faster than release", () => {
    const frames = manualScheduler();
    const smoother = new LevelSmoother(() => {}, frames.scheduler);

    smoother.push(1.0);
    const afterAttack = smoother.value;
    expect(afterAttack).toBeCloseTo(0.35, 5);

    smoother.push(0.0);
    expect(smoother.value).toBeCloseTo(afterAttack * 0.85, 5);
  });

  test("reset drops the pending frame so no stale commit fires", () => {
    const frames = manualScheduler();
    const commits: number[] = [];
    const smoother = new LevelSmoother((v) => commits.push(v), frames.scheduler);

    smoother.push(0.8);
    smoother.reset();
    frames.flush();

    expect(commits.length).toBe(0);
    expect(smoother.value).toBe(0);
  });
});

describe("dotStyle", () => {
  test("live level disables the idle keyframe animation", () => {
    // Regression guard: the dot-bounce animation beats inline styles in the
    // CSS cascade. If `animation: none` is dropped, the level-driven
    // transform never renders and the dots freeze in the idle loop.
    const style = dotStyle(0.9, 3); // index 3 = centre bar, weight 1.0
    expect(style.animation).toBe("none");
    expect(style.transform).toBe(`scaleY(${1 + 0.9 * 1.0 * 2.0})`);
    expect(style.opacity).toBeCloseTo(0.55 + 0.9 * 0.45, 5);
  });

  test("edge dots react less than the center dot", () => {
    const center = dotStyle(1.0, 3);
    const edge = dotStyle(1.0, 0);
    const scaleOf = (s: React.CSSProperties) =>
      Number(/scaleY\(([^)]+)\)/.exec(String(s.transform))?.[1]);
    expect(scaleOf(edge)).toBeLessThan(scaleOf(center));
  });

  test("no level keeps the staggered idle bounce", () => {
    const style = dotStyle(undefined, 3);
    expect(style.animation).toBeUndefined();
    expect(style.transform).toBeUndefined();
    // Index 3 is a centre bar, so it leads the ripple with no delay.
    expect(style.animationDelay).toBe("0ms");
  });

  test("zero level keeps the idle bounce (silence before first event)", () => {
    const style = dotStyle(0, 2);
    expect(style.animationDelay).toBe("110ms");
    expect(style.animation).toBeUndefined();
  });

  /** The idle ripple runs centre-out, matching the centre-weighted shape the
   *  live level produces — a left-to-right ramp would make idle and speaking
   *  look like two unrelated widgets. */
  test("idle delays are symmetric about the centre", () => {
    const delays = Array.from({ length: DOT_COUNT }, (_, i) => dotStyle(undefined, i).animationDelay);
    expect(delays).toEqual([...delays].reverse());
    expect(delays).toEqual(["330ms", "220ms", "110ms", "0ms", "110ms", "220ms", "330ms"]);
  });

  test("idle delay grows with distance from the centre", () => {
    const centre = Number.parseInt(String(dotStyle(undefined, 2).animationDelay), 10);
    const mid = Number.parseInt(String(dotStyle(undefined, 1).animationDelay), 10);
    const edge = Number.parseInt(String(dotStyle(undefined, 0).animationDelay), 10);
    expect(centre).toBeLessThan(mid);
    expect(mid).toBeLessThan(edge);
  });
});
