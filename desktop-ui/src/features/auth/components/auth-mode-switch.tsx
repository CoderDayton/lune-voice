import { useStore } from "@tanstack/react-store";
import { useHotkey } from "@tanstack/react-hotkeys";
import { Button } from "../../../components/ui/button";
import { authActions, authStore, type AuthMode } from "../../../store/auth-store";

const authModes: Array<{ value: AuthMode; label: string }> = [
  { value: "login", label: "Log in" },
  { value: "signup", label: "Sign up" },
];

export function AuthModeSwitch() {
  const mode = useStore(authStore, (state) => state.mode);

  useHotkey("Alt+1", () => authActions.setMode("login"));
  useHotkey("Alt+2", () => authActions.setMode("signup"));

  return (
    <div className="auth-mode-switch">
      {authModes.map((option) => (
        <Button
          key={option.value}
          type="button"
          variant={mode === option.value ? "primary" : "ghost"}
          className="auth-mode-button"
          onClick={() => authActions.setMode(option.value)}
        >
          {option.label}
        </Button>
      ))}
    </div>
  );
}
