import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/cn";

type PanelTone = "default" | "soft" | "hero";

const toneClasses: Record<PanelTone, string> = {
  default: "card bg-base-200/80 border border-base-300/40",
  soft: "card bg-base-300/50 border border-base-300/30",
  hero: "card bg-base-200/80 border border-base-300/60 surface-hero",
};

interface PanelProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  tone?: PanelTone;
}

export function Panel({ children, className, tone = "default", ...props }: PanelProps) {
  return (
    <div className={cn(toneClasses[tone], className)} {...props}>
      {children}
    </div>
  );
}
