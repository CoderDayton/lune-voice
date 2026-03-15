import type { ReactNode } from "react";
import { cn } from "../../lib/cn";
import { Panel } from "./panel";

interface PageIntroPanelProps {
  kicker: string;
  title: string;
  description: string;
  className?: string;
  children?: ReactNode;
}

export function PageIntroPanel({
  kicker,
  title,
  description,
  className,
  children,
}: PageIntroPanelProps) {
  return (
    <Panel className={cn("p-4 sm:p-6", className)}>
      <div className="page-intro sm:max-w-[var(--size-measure-wide)]">
        <p className="app-kicker-accent">{kicker}</p>
        <h2 className="page-title mt-3">{title}</h2>
        <p className="body-copy mt-3">{description}</p>
        {children}
      </div>
    </Panel>
  );
}
