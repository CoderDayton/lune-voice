// SPDX-License-Identifier: GPL-3.0-or-later

import { Check, ChevronDown } from 'lucide-react';
import {
  type ComponentType,
  type CSSProperties,
  type ReactNode,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';

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

/** Tallest the open menu grows before it scrolls, in pixels. */
const MENU_MAX_HEIGHT = 360;

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
  const [menuStyle, setMenuStyle] = useState<CSSProperties | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const listboxId = useId();

  const current = options.find((opt) => opt.value === value) ?? options[0];
  // Primitives, not `options` itself: callers build that array on every render,
  // and an effect keyed on it re-ran with each scroll and snapped the list
  // back to the selected row.
  const optionCount = options.length;
  const hasDescriptions = options.some((opt) => opt.description);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      const target = event.target as Node;
      if (!containerRef.current?.contains(target) && !menuRef.current?.contains(target)) {
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

  useEffect(() => {
    if (!open) return;

    function updateMenuPosition(event?: Event) {
      // Scrolling the menu itself must not reposition it.
      if (event?.target instanceof Node && menuRef.current?.contains(event.target)) return;
      const trigger = containerRef.current;
      if (!trigger) return;

      const rect = trigger.getBoundingClientRect();
      const gutter = 8;
      const width = Math.min(320, window.innerWidth - gutter * 2);
      const left =
        align === 'end'
          ? Math.min(Math.max(gutter, rect.right - width), window.innerWidth - width - gutter)
          : Math.min(Math.max(gutter, rect.left), window.innerWidth - width - gutter);

      // Flip up when there isn't enough room below the trigger and there's
      // more room above — otherwise the menu gets clamped tiny or runs off the
      // bottom of the screen.
      const spaceBelow = window.innerHeight - rect.bottom - gutter;
      const spaceAbove = rect.top - gutter;
      // Prefer the measured height once the portal is mounted; on first open it
      // isn't rendered yet, so estimate from the option list.
      const estRow = hasDescriptions ? 60 : 44;
      const estContent = optionCount * estRow + 12;
      const desired = menuRef.current?.scrollHeight || estContent;
      // About eight plain rows (fewer with descriptions): a long list such as
      // the dictation languages scrolls inside the menu instead of covering
      // most of the window. Still never more than 60% of a short window.
      const cap = Math.min(MENU_MAX_HEIGHT, Math.round(window.innerHeight * 0.6));
      const needed = Math.min(desired, cap);
      const openUp = spaceBelow < needed && spaceAbove > spaceBelow;
      const available = openUp ? spaceAbove : spaceBelow;
      const maxHeight = Math.max(160, Math.min(cap, available));

      setMenuStyle({
        position: 'fixed',
        zIndex: 'var(--z-popover, 400)',
        ...(openUp
          ? { bottom: window.innerHeight - rect.top + gutter }
          : { top: rect.bottom + gutter }),
        left,
        width,
        maxHeight: `${maxHeight}px`,
      });
    }

    updateMenuPosition();
    window.addEventListener('resize', updateMenuPosition);
    window.addEventListener('scroll', updateMenuPosition, true);
    return () => {
      window.removeEventListener('resize', updateMenuPosition);
      window.removeEventListener('scroll', updateMenuPosition, true);
    };
  }, [open, align, optionCount, hasDescriptions]);

  // Once, on open: a choice far down a scrolling list is visible straight away.
  useEffect(() => {
    if (!open) return;
    const reveal = requestAnimationFrame(() => {
      menuRef.current
        ?.querySelector<HTMLElement>('[aria-selected="true"]')
        ?.scrollIntoView({ block: 'nearest' });
    });
    return () => cancelAnimationFrame(reveal);
  }, [open]);

  function pick(next: Value) {
    onChange(next);
    setOpen(false);
  }

  const TriggerIcon = current?.icon;
  const menu =
    open && menuStyle
      ? createPortal(
          <div
            ref={menuRef}
            id={listboxId}
            role="listbox"
            aria-label={ariaLabel}
            style={menuStyle}
            className="flex flex-col gap-1 overflow-y-auto overscroll-contain rounded-box border border-surface-border-strong bg-surface-elevated p-1.5 shadow-overlay"
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
                  className={`group flex w-full items-start gap-3 rounded-field px-2.5 py-2 text-start transition-colors duration-fast hover:bg-surface-hover ${
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
          </div>,
          document.body,
        )
      : null;

  return (
    <div ref={containerRef} className={`relative inline-block ${className}`}>
      <button
        type="button"
        className={`field-trigger w-full sm:w-72 ${buttonClassName}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        aria-label={ariaLabel}
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="flex min-w-0 items-center gap-2.5 text-start">
          {TriggerIcon && (
            <span className="field-chip flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/15 text-primary">
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
      {menu}
    </div>
  );
}
