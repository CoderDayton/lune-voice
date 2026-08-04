/** Pill overlay icons — stroke-based, 24×24 viewBox, inherits currentColor. */

interface IconProps {
  className?: string;
}

function PillSvg({ className = 'size-5', children }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

/** Discard. */
export function XIcon({ className }: IconProps = {}) {
  return (
    <PillSvg className={className ?? 'size-[15px]'}>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </PillSvg>
  );
}

/** Accept — stop and transcribe. An arrow rather than a tick: the utterance is
 *  being sent somewhere (the focused app), not merely approved. */
export function ArrowUpIcon({ className }: IconProps = {}) {
  return (
    <PillSvg className={className ?? 'size-[15px]'}>
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </PillSvg>
  );
}

export function PulseIndicator({ label }: { label: string }) {
  return <span className="pill-pulse" role="img" aria-label={label} />;
}
