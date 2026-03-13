import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function BaseIcon({ children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 18 18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

export function OverviewIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <rect x="2" y="2" width="5.2" height="5.2" rx="1.2" />
      <rect x="10.8" y="2" width="5.2" height="5.2" rx="1.2" />
      <rect x="2" y="10.8" width="5.2" height="5.2" rx="1.2" />
      <path d="M11 13h5M13.5 10.5v5" />
    </BaseIcon>
  );
}

export function HistoryIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <circle cx="9" cy="9" r="6.5" />
      <path d="M9 5.3v4l2.8 2" />
    </BaseIcon>
  );
}

export function DictionaryIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M4 3.5h9.5v11H4z" />
      <path d="M6.2 6h5M6.2 8.9h5M6.2 11.8h3.4" />
    </BaseIcon>
  );
}

export function SearchIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <circle cx="7.5" cy="7.5" r="4.5" />
      <path d="m11 11 4 4" />
    </BaseIcon>
  );
}

export function ChevronLeftIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="m11 3.5-5 5 5 5" />
    </BaseIcon>
  );
}

export function ChevronRightIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="m7 3.5 5 5-5 5" />
    </BaseIcon>
  );
}

export function BoltIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="m9.6 2.5-4.2 6h2.6l-.9 7 4.4-6.3H8.8z" />
    </BaseIcon>
  );
}

export function ClockIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <circle cx="9" cy="9" r="6.4" />
      <path d="M9 5.2v4l2.4 1.6" />
    </BaseIcon>
  );
}

export function MicIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <rect x="6.3" y="2.8" width="5.4" height="8.8" rx="2.7" />
      <path d="M4.5 8.8v.5a4.5 4.5 0 0 0 9 0v-.5M9 13.8v2.2M6.8 16h4.4" />
    </BaseIcon>
  );
}

export function PulseIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M2.2 9h3l1.4-3.4L9.4 12l2-4h4.4" />
    </BaseIcon>
  );
}

export function MenuIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M3 5h12M3 9h12M3 13h12" />
    </BaseIcon>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="m4.5 4.5 9 9M13.5 4.5l-9 9" />
    </BaseIcon>
  );
}

export function LuneLogo(props: IconProps) {
  return (
    <svg width="20" height="20" viewBox="0 0 22 22" fill="none" aria-hidden="true" {...props}>
      <path
        d="M4.5 16.5V5.5l6 7.5 6-7.5v11"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
