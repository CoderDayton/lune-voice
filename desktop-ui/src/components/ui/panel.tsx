import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/cn";

type PanelTone = "default" | "soft" | "hero";

const toneClasses: Record<PanelTone, string> = {
  default: "surface-panel",
  soft: "surface-soft",
  hero: "surface-hero",
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
