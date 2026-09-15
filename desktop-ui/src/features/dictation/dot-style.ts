// SPDX-License-Identifier: GPL-3.0-or-later

export const DOT_COUNT = 7;

// Per-bar scale offsets — center bars react most, edges least (symmetric). An
// odd count so the arch has a single peak: an even one puts two equal bars at
// the center and flattens the top of every shape the meter makes.
const DOT_WEIGHTS = [0.35, 0.6, 0.85, 1.0, 0.85, 0.6, 0.35];

// Idle-bounce delay between one symmetric pair of bars and the next.
const DOT_STAGGER_MS = 110;

/**
 * Idle-bounce delay for one bar: bars equidistant from the center share a
 * delay, so the resting animation ripples outward from the middle.
 *
 * This mirrors `DOT_WEIGHTS`, which already treats the strip as symmetric about
 * its center. A plain `index * delay` ramp reads as a marquee travelling left to
 * right, which fights the center-weighted shape the live level produces — idle
 * and speaking then look like two unrelated widgets. Works for an even or odd
 * `DOT_COUNT`: the floor collapses the half-step an even count puts on center.
 */
function idleDelayMs(index: number): number {
  const center = (DOT_COUNT - 1) / 2;
  return Math.floor(Math.abs(index - center)) * DOT_STAGGER_MS;
}

/** Resting height of the centre bars, in px. Mirrors `DOT_H` in pill_paint.rs. */
const DOT_HEIGHT_PX = 11;

/** Opacity floor, shared with the processing pulse: below it the mark all but
 *  vanishes on the lightened bar. Mirrors `DOT_MIN_ALPHA` in pill_paint.rs. */
const DOT_MIN_ALPHA = 0.55;

/**
 * Resting height of one bar: full height at the centre, tapering to half the
 * taper at the edges. The same weight drives height and response, so a bar that
 * reacts less also sits lower — which is what keeps the band reading as a
 * waveform in near-silence, where the smoothed level sits just above zero and a
 * uniform height would render a flat picket fence.
 */
function barHeightPx(weight: number): number {
  return (DOT_HEIGHT_PX * (1 + weight)) / 2;
}

/**
 * Inline style for one audio dot.
 *
 * With a live level the dot is driven by transform/opacity, and the idle
 * `dot-bounce` keyframe animation MUST be switched off inline: CSS animations
 * beat inline declarations in the cascade, so leaving it running would
 * override the transform and freeze the dots in the idle loop during speech
 * (the pre-fix Linux "stuck pill" symptom). Without a level, the dots run the
 * staggered idle bounce.
 */
export function dotStyle(level: number | undefined, index: number): React.CSSProperties {
  const weight = DOT_WEIGHTS[index] ?? 0;
  const height = `${barHeightPx(weight)}px`;
  const hasLevel = level !== undefined && level > 0;
  if (!hasLevel) {
    return { height, animationDelay: `${idleDelayMs(index)}ms` };
  }
  return {
    height,
    transform: `scaleY(${1 + level * weight * 2.0})`,
    opacity: DOT_MIN_ALPHA + level * (1 - DOT_MIN_ALPHA),
    animation: 'none',
  };
}
