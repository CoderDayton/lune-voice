interface ToggleProps {
  checked: boolean;
  onChange: () => void;
}

export function Toggle({ checked, onChange }: ToggleProps) {
  return (
    <input
      type="checkbox"
      className="toggle toggle-primary"
      checked={checked}
      onChange={onChange}
    />
  );
}
