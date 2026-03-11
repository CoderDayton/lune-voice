import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen gap-4 p-4 sm:gap-6 sm:p-8 md:p-12 lg:p-16">
      <h1 className="text-3xl font-bold tracking-tight text-[var(--color-text)] sm:text-4xl md:text-5xl">
        Lune Voice
      </h1>
      <p className="text-base text-[var(--color-text-muted)] sm:text-lg md:text-xl">
        Voice dictation for Linux
      </p>
    </main>
  );
}
