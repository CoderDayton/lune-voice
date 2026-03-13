import { useStore } from "@tanstack/react-store";
import { useHotkey } from "@tanstack/react-hotkeys";
import { cn } from "../../../lib/cn";
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
    <div className="join">
      {authModes.map((option) => (
        <button
          key={option.value}
          type="button"
          className={cn(
            "join-item btn btn-sm min-w-24",
            mode === option.value ? "btn-primary" : "btn-ghost",
          )}
          onClick={() => authActions.setMode(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
