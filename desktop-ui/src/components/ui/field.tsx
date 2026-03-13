import type { InputHTMLAttributes, SelectHTMLAttributes } from "react";
import { cn } from "../../lib/cn";

export function TextField({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn("input input-bordered w-full", className)} {...props} />;
}

export function SelectField({ className, children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn("select select-bordered w-full", className)} {...props}>
      {children}
    </select>
  );
}
