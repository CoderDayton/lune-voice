const DOT_COUNT = 5;

// Per-dot scale offsets — center dot reacts most, edges less
const DOT_WEIGHTS = [0.5, 0.8, 1.0, 0.8, 0.5];

interface AudioDotsProps {
  level?: number; // 0.0–1.0 from real audio RMS
}

export function AudioDots({ level }: AudioDotsProps) {
  const hasLevel = level !== undefined && level > 0;

  return (
    <div className="audio-dots" role="img" aria-label="Audio activity">
      {Array.from({ length: DOT_COUNT }, (_, i) => {
        const scale = hasLevel
          ? 1 + level * DOT_WEIGHTS[i]! * 2.0
          : undefined;

        return (
          <span
            key={i}
            className="audio-dot"
            style={
              hasLevel
                ? { transform: `scaleY(${scale})`, opacity: 0.4 + level * 0.6 }
                : { animationDelay: `${i * 120}ms` }
            }
          />
        );
      })}
    </div>
  );
}
