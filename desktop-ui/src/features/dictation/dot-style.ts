export const DOT_COUNT = 6;

// Per-bar scale offsets — center bars react most, edges least (symmetric).
const DOT_WEIGHTS = [0.5, 0.75, 1.0, 1.0, 0.75, 0.5];

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
  const hasLevel = level !== undefined && level > 0;
  if (!hasLevel) {
    return { animationDelay: `${index * 120}ms` };
  }
  const weight = DOT_WEIGHTS[index] ?? 0;
  return {
    transform: `scaleY(${1 + level * weight * 2.0})`,
    opacity: 0.4 + level * 0.6,
    animation: 'none',
  };
}
