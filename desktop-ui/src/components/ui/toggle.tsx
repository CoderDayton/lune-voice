import { cn } from "../../lib/cn";

interface ToggleProps {
  checked: boolean;
  onChange: () => void;
}

export function Toggle({ checked, onChange }: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      className={cn(checked ? "toggle-track-on" : "toggle-track-off")}
    >
      <span className={cn(checked ? "toggle-knob-on" : "toggle-knob-off")} />
    </button>
  );
}
