import { useStore } from "@tanstack/react-store";
import { AuthScreen } from "../../auth/components/auth-screen";
import { AppShell } from "../../shell/components/app-shell";
import { authStore, isAuthenticated } from "../../../store/auth-store";

export function DesktopRoot() {
  const authenticated = useStore(authStore, isAuthenticated);

  if (!authenticated) {
    return <AuthScreen />;
  }

  return <AppShell />;
}
