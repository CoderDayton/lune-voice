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
    <Panel className={cn("panel-pad-lg", className)}>
      <div className="page-intro">
        <p className="app-kicker-accent">{kicker}</p>
        <h2 className="page-title mt-3">{title}</h2>
        <p className="body-copy mt-3">{description}</p>
        {children}
      </div>
    </Panel>
  );
}
