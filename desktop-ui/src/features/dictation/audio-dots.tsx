const DOT_COUNT = 5;

export function AudioDots() {
  return (
    <div className="audio-dots" role="img" aria-label="Audio activity">
      {Array.from({ length: DOT_COUNT }, (_, i) => (
        <span
          key={i}
          className="audio-dot"
          style={{ animationDelay: `${i * 120}ms` }}
        />
      ))}
    </div>
  );
}
