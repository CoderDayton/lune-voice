import { TextField } from "../../../components/ui/field";

interface AuthInputFieldProps {
  label: string;
  type?: "text" | "email" | "password";
  placeholder: string;
  value: string;
  error: string | null;
  onChange: (value: string) => void;
  onBlur: () => void;
}

export function AuthInputField({
  label,
  type = "text",
  placeholder,
  value,
  error,
  onChange,
  onBlur,
}: AuthInputFieldProps) {
  return (
    <label className="stack-copy-tight">
      <span className="label-text-custom block mb-2">{label}</span>
      <TextField
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onBlur={onBlur}
        placeholder={placeholder}
        className={error ? "input-error" : undefined}
      />
      {error ? <span className="text-error text-xs">{error}</span> : null}
    </label>
  );
}
