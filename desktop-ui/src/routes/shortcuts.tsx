import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/shortcuts")({
  beforeLoad: () => {
    throw redirect({ to: "/settings" });
  },
});
