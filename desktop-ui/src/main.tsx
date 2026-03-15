import { HotkeysProvider } from "@tanstack/react-hotkeys";
import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { routeTree } from "./routeTree.gen";
import "./app.css";
import "./styles/shell.css";
import "./styles/typography.css";
import "./styles/topbar.css";
import "./styles/sidebar.css";
import "./styles/auth.css";
import "./styles/history.css";

const queryClient = new QueryClient();
const router = createRouter({ routeTree, context: { queryClient } });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <HotkeysProvider>
        <RouterProvider router={router} />
      </HotkeysProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
