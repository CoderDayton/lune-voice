import { type ReactNode, useLayoutEffect, useRef, useState } from 'react';

/**
 * Unfolds its children in place, growing from zero height.
 *
 * For a control that is *replaced* by the panel it opens — Snippets' New
 * snippet bar becoming the composer. Swapped instantly, the panel reads as a
 * box appearing from nowhere above whatever you clicked; grown from the spot
 * the control occupied, it reads as that control opening.
 *
 * WAAPI rather than a CSS transition: `height: auto` is not interpolable, and
 * starting each run from the in-flight height lets a mid-motion reverse turn
 * around smoothly instead of jumping. Children stay mounted through the
 * fold-up so closing can finish animating before they leave the DOM.
 *
 * Durations and easings match the Sessions row expander, so the two folds in
 * the app feel like one motion.
 */
export function Fold({
  open,
  className,
  children,
}: {
  open: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [shown, setShown] = useState(open);
  const ref = useRef<HTMLDivElement>(null);
  const animRef = useRef<Animation | null>(null);

  useLayoutEffect(() => {
    if (open && !shown) {
      setShown(true); // mount first; the next pass runs the unfold
      return;
    }
    const el = ref.current;
    if (!shown || !el) return;

    const midFlight = animRef.current?.playState === 'running';
    const fromHeight = midFlight ? el.getBoundingClientRect().height : open ? 0 : el.offsetHeight;
    const fromOpacity = midFlight ? Number(getComputedStyle(el).opacity) : open ? 0 : 1;
    animRef.current?.cancel();

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const root = getComputedStyle(document.documentElement);
    const ease = open ? '--ease-decelerate' : '--ease-accelerate';

    const anim = el.animate(
      [
        { height: `${fromHeight}px`, opacity: fromOpacity },
        open ? { height: `${el.offsetHeight}px`, opacity: 1 } : { height: '0px', opacity: 0 },
      ],
      {
        duration: reduced ? 0 : open ? 260 : 190,
        easing: root.getPropertyValue(ease).trim() || (open ? 'ease-out' : 'ease-in'),
        // Hold the collapsed frame until the unmount lands, or the panel
        // flashes back to full height for one frame on the way out.
        fill: open ? 'none' : 'forwards',
      },
    );
    animRef.current = anim;
    if (!open) anim.onfinish = () => setShown(false);
  }, [open, shown]);

  if (!shown) return null;

  return (
    <div ref={ref} className={`overflow-hidden ${className ?? ''}`}>
      {children}
    </div>
  );
}
