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
      strokeWidth={2}
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

export function MicIcon({ className }: IconProps = {}) {
  return (
    <PillSvg className={className ?? 'size-5'}>
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 10v1a7 7 0 0 0 14 0v-1" />
      <line x1="12" y1="18" x2="12" y2="22" />
      <line x1="9" y1="22" x2="15" y2="22" />
    </PillSvg>
  );
}

/** Brand mark — a filled crescent (Lune). Solid fill so it reads as a logo at
 *  16px, where the stroke icons above would look wiry. */
export function MoonIcon({ className }: IconProps = {}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className ?? 'size-4'}
      aria-hidden="true"
      focusable="false"
    >
      <path d="M12 3a6.364 6.364 0 0 0 9 9 9 9 0 1 1-9-9Z" />
    </svg>
  );
}

export function XIcon({ className }: IconProps = {}) {
  return (
    <PillSvg className={className ?? 'size-4'}>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </PillSvg>
  );
}

export function StopIcon({ className }: IconProps = {}) {
  return (
    <PillSvg className={className ?? 'size-4'}>
      <rect x="6" y="6" width="12" height="12" rx="1" />
    </PillSvg>
  );
}

export function PulseIndicator({ label }: { label: string }) {
  return <span className="pill-pulse" role="img" aria-label={label} />;
}
