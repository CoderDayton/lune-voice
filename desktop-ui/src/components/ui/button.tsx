import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/cn";

const buttonVariants = {
  primary: "btn btn-primary",
  secondary: "btn btn-ghost",
  ghost: "btn btn-ghost btn-sm",
  danger: "btn btn-error",
  table: "btn btn-ghost btn-sm",
  sidebar: "btn btn-ghost w-full justify-start",
} as const;

type ButtonVariant = keyof typeof buttonVariants;

interface CommonProps {
  children: ReactNode;
  className?: string;
  variant?: ButtonVariant;
}

type ButtonProps = CommonProps & ButtonHTMLAttributes<HTMLButtonElement>;
type AnchorProps = CommonProps & AnchorHTMLAttributes<HTMLAnchorElement>;

export function buttonClassName(variant: ButtonVariant = "primary", className?: string) {
  return cn(buttonVariants[variant], className);
}

export function Button({ children, className, variant = "primary", ...props }: ButtonProps) {
  return (
    <button className={buttonClassName(variant, className)} {...props}>
      {children}
    </button>
  );
}

export function AnchorButton({ children, className, variant = "primary", ...props }: AnchorProps) {
  return (
    <a className={buttonClassName(variant, className)} {...props}>
      {children}
    </a>
  );
}
