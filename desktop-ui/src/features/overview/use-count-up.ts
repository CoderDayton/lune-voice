import { useEffect, useRef, useState } from 'react';

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;

/**
 * Animates a number from its currently displayed value up (or down) to
 * `target` over `duration` ms using requestAnimationFrame and an easeOutCubic
 * curve. Honors `prefers-reduced-motion` by snapping straight to the target.
 *
 * Mid-flight target changes restart the tween from whatever is on screen, so
 * live-updating stats never jump.
 */
export function useCountUp(target: number, duration = 600): number {
  // Seed at 0 so the first mount animates up to the target (a load-in
  // count). Reduced-motion / zero-duration users start at the target to
  // avoid a one-frame flash of 0.
  const [value, setValue] = useState(() => (duration <= 0 || prefersReducedMotion() ? target : 0));
  // Mirror the latest rendered value so a new tween starts from what the user
  // currently sees, not a stale closure capture.
  const valueRef = useRef(value);
  valueRef.current = value;
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    if (duration <= 0 || prefersReducedMotion()) {
      setValue(target);
      return;
    }
    const from = valueRef.current;
    if (from === target) return;

    let start = 0;
    const tick = (now: number) => {
      if (start === 0) start = now;
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - t) ** 3;
      setValue(Math.round(from + (target - from) * eased));
      if (t < 1) frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    };
  }, [target, duration]);

  return value;
}
