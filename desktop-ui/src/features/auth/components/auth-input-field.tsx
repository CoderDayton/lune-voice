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
      <span className="field-label">{label}</span>
      <TextField
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onBlur={onBlur}
        placeholder={placeholder}
        className={error ? "auth-field-error" : undefined}
      />
      {error ? <span className="auth-field-error-copy">{error}</span> : null}
    </label>
  );
}
