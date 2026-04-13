import { createRootRoute } from "@tanstack/react-router";
import { AppShell } from "../features/shell/components/app-shell";

export const Route = createRootRoute({
  component: AppShell,
});
