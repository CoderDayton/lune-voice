interface ToggleProps {
  checked: boolean;
  onChange: () => void;
  "aria-label"?: string;
}

export function Toggle({ checked, onChange, "aria-label": ariaLabel }: ToggleProps) {
  return (
    <input
      type="checkbox"
      className="toggle toggle-primary"
      checked={checked}
      onChange={onChange}
      aria-label={ariaLabel}
    />
  );
}
