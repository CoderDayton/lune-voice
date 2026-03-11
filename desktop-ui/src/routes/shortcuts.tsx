import { createFileRoute } from "@tanstack/react-router";
import { ShortcutsPage } from "../features/shortcuts/shortcuts-page";

export const Route = createFileRoute("/shortcuts")({
  component: ShortcutsPage,
});
