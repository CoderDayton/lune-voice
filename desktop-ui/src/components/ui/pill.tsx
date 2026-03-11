import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/cn";

interface PillProps extends HTMLAttributes<HTMLSpanElement> {
  children: ReactNode;
}

export function Pill({ children, className, ...props }: PillProps) {
  return (
    <span className={cn("app-pill", className)} {...props}>
      {children}
    </span>
  );
}
