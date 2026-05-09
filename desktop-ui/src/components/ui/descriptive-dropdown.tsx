import { Check, ChevronDown } from 'lucide-react';
import { type ComponentType, type ReactNode, useEffect, useId, useRef, useState } from 'react';

type IconComponent = ComponentType<{
  size?: number;
  strokeWidth?: number;
  className?: string;
}>;

export interface DescriptiveOption<Value extends string> {
  value: Value;
  label: string;
  description?: string;
  icon?: IconComponent;
  badge?: ReactNode;
}

interface DescriptiveDropdownProps<Value extends string> {
  value: Value;
  options: DescriptiveOption<Value>[];
  onChange: (next: Value) => void;
  ariaLabel?: string;
  className?: string;
  buttonClassName?: string;
  align?: 'start' | 'end';
}

/**
 * DaisyUI-styled dropdown that renders icon + label + description per option.
 * Drop-in replacement for `<select>` when option meaning isn't obvious from
 * the label alone.
 */
export function DescriptiveDropdown<Value extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  className = '',
  buttonClassName = '',
  align = 'end',
}: DescriptiveDropdownProps<Value>) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const listboxId = useId();

  const current = options.find((opt) => opt.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  function pick(next: Value) {
    onChange(next);
    setOpen(false);
  }

  const TriggerIcon = current?.icon;
  const menuPos = align === 'end' ? 'right-0' : 'left-0';

  return (
    <div ref={containerRef} className={`relative inline-block ${className}`}>
      <button
        type="button"
        className={`btn justify-between gap-3 w-full sm:w-72 normal-case font-normal border border-surface-border bg-surface-elevated hover:bg-surface-hover ${buttonClassName}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        aria-label={ariaLabel}
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="flex min-w-0 items-center gap-2.5 text-left">
          {TriggerIcon && (
            <span className="flex size-8 shrink-0 items-center justify-center rounded-selector bg-primary/15 text-primary">
              <TriggerIcon size={16} strokeWidth={1.8} />
            </span>
          )}
          <span className="truncate text-content">{current?.label}</span>
        </span>
        <ChevronDown
          size={16}
          strokeWidth={1.8}
          className={`shrink-0 text-content-subtle transition-transform duration-fast ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div
          id={listboxId}
          role="listbox"
          aria-label={ariaLabel}
          className={`absolute ${menuPos} z-dropdown mt-2 flex max-h-[min(60vh,28rem)] w-[min(20rem,90vw)] flex-col gap-1 overflow-y-auto rounded-box border border-surface-border bg-surface-elevated p-1.5 shadow-card backdrop-blur-xl`}
        >
          {options.map((option) => {
            const Icon = option.icon;
            const active = option.value === value;
            return (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={active}
                onClick={() => pick(option.value)}
                className={`group flex w-full items-start gap-3 rounded-field px-2.5 py-2 text-left transition-colors duration-fast hover:bg-surface-hover ${
                  active ? 'bg-primary/8' : ''
                }`}
              >
                {Icon && (
                  <span
                    className={`mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-selector ${
                      active
                        ? 'bg-primary/15 text-primary'
                        : 'bg-surface-hover text-content-subtle group-hover:text-content'
                    }`}
                  >
                    <Icon size={16} strokeWidth={1.8} />
                  </span>
                )}
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="flex items-center gap-2">
                    <span
                      className={`text-sm font-medium ${active ? 'text-content-strong' : 'text-content'}`}
                    >
                      {option.label}
                    </span>
                    {option.badge}
                  </span>
                  {option.description && (
                    <span className="text-xs leading-snug text-content-subtle">
                      {option.description}
                    </span>
                  )}
                </span>
                {active && (
                  <Check size={14} strokeWidth={2.5} className="mt-1.5 shrink-0 text-primary" />
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
