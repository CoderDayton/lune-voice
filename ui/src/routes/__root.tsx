import { Outlet, createRootRouteWithContext } from "@tanstack/react-router";
import type { QueryClient } from "@tanstack/react-query";

interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="min-h-screen bg-[var(--color-bg)] px-4 sm:px-6 md:px-8 lg:px-12">
      <Outlet />
    </div>
  );
}
