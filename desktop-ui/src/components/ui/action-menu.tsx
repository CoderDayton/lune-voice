// SPDX-License-Identifier: GPL-3.0-or-later

import {
  type ComponentType,
  type CSSProperties,
  type RefObject,
  useCallback,
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

export interface ActionMenuItem {
  id: string;
  label: string;
  icon?: IconComponent;
  /** Right-aligned hint — a shortcut like `⌘K`, or a current value like `Dark`. */
  trailing?: string;
  /** Renders in the error colour. Reserved for destructive actions. */
  danger?: boolean;
  /** Draws a hairline above this item, grouping what follows. */
  separatorBefore?: boolean;
  onSelect: () => void;
}

/** `top` stacks above the anchor; `right` flanks it (the collapsed rail). */
type Placement = 'top' | 'right';

interface ActionMenuProps {
  items: ActionMenuItem[];
  ariaLabel: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Element the menu positions against, and returns focus to on close. */
  anchorRef: RefObject<HTMLElement | null>;
  placement?: Placement;
  /** Fixed width for `right`. `top` matches the anchor instead. */
  width?: number;
}

const GUTTER = 8;
/** Matches the mock: the expanded menu insets from the sidebar's own edges. */
const ANCHOR_INSET = 4;
/** Floor for the anchor-matched width, so a narrow sidebar stays readable. */
const MIN_WIDTH = 180;

function computeMenuStyle(rect: DOMRect, placement: Placement, width: number): CSSProperties {
  const maxWidth = window.innerWidth - GUTTER * 2;

  if (placement === 'right') {
    const menuWidth = Math.min(width, maxWidth);
    return {
      position: 'fixed',
      zIndex: 'var(--z-popover, 400)',
      left: Math.min(rect.right + GUTTER, window.innerWidth - menuWidth - GUTTER),
      // Bottom-aligned to the anchor, so the menu grows upward like the mock.
      bottom: Math.max(GUTTER, window.innerHeight - rect.bottom),
      width: menuWidth,
      maxHeight: `${window.innerHeight - GUTTER * 2}px`,
    };
  }

  const menuWidth = Math.min(Math.max(rect.width - ANCHOR_INSET * 2, MIN_WIDTH), maxWidth);
  return {
    position: 'fixed',
    zIndex: 'var(--z-popover, 400)',
    bottom: window.innerHeight - rect.top + GUTTER,
    left: Math.min(
      Math.max(GUTTER, rect.left + ANCHOR_INSET),
      window.innerWidth - menuWidth - GUTTER,
    ),
    width: menuWidth,
    maxHeight: `${Math.max(160, rect.top - GUTTER * 2)}px`,
  };
}

/** Tracks the anchor's box while open, re-measuring on resize and scroll. */
function useMenuPosition(
  open: boolean,
  anchorRef: RefObject<HTMLElement | null>,
  placement: Placement,
  width: number,
) {
  const [menuStyle, setMenuStyle] = useState<CSSProperties | null>(null);

  useEffect(() => {
    if (!open) {
      setMenuStyle(null);
      return;
    }
    function updatePosition() {
      const anchor = anchorRef.current;
      if (!anchor) return;
      setMenuStyle(computeMenuStyle(anchor.getBoundingClientRect(), placement, width));
    }
    updatePosition();
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
    };
  }, [open, placement, width, anchorRef]);

  return menuStyle;
}

function ActionMenuRow({
  item,
  itemRef,
  onActivate,
}: {
  item: ActionMenuItem;
  itemRef: (el: HTMLButtonElement | null) => void;
  onActivate: () => void;
}) {
  const Icon = item.icon;
  return (
    <>
      {item.separatorBefore && (
        <div aria-hidden="true" className="my-1 border-t border-surface-border" />
      )}
      <button
        ref={itemRef}
        type="button"
        role="menuitem"
        onClick={onActivate}
        className={`group flex h-10 w-full shrink-0 items-center gap-3 rounded-full pe-3 ps-1.5 text-start text-sm transition-colors duration-fast focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary ${
          item.danger
            ? 'text-error hover:bg-error/10 focus-visible:bg-error/10'
            : 'text-content hover:bg-surface-hover focus-visible:bg-surface-hover'
        }`}
      >
        {Icon && (
          <span
            className={`flex size-7 shrink-0 items-center justify-center rounded-full ${
              item.danger ? 'bg-error/10' : 'bg-surface-wash'
            }`}
          >
            <Icon
              size={15}
              strokeWidth={1.8}
              className={item.danger ? '' : 'text-content-muted group-hover:text-content'}
            />
          </span>
        )}
        <span className="min-w-0 flex-1 truncate">{item.label}</span>
        {item.trailing && (
          <kbd
            className={`shrink-0 font-mono text-xs ${
              item.danger ? 'text-error/70' : 'text-content-subtle'
            }`}
          >
            {item.trailing}
          </kbd>
        )}
      </button>
    </>
  );
}

/**
 * A menu of actions, as opposed to `DescriptiveDropdown`, which selects a value.
 * Controlled, and positioned against an arbitrary anchor so the sidebar can open
 * it upward when expanded and sideways from the collapsed rail.
 */
export function ActionMenu({
  items,
  ariaLabel,
  open,
  onOpenChange,
  anchorRef,
  placement = 'top',
  width = 240,
}: ActionMenuProps) {
  const menuRef = useRef<HTMLDivElement | null>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const menuId = useId();
  const menuStyle = useMenuPosition(open, anchorRef, placement, width);

  const close = useCallback(
    (returnFocus: boolean) => {
      onOpenChange(false);
      if (returnFocus) anchorRef.current?.focus();
    },
    [onOpenChange, anchorRef],
  );

  // Dismiss on outside press. Pointerdown rather than click so a press that
  // starts outside never lands on whatever the menu was covering.
  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      const target = event.target as Node;
      if (!anchorRef.current?.contains(target) && !menuRef.current?.contains(target)) {
        onOpenChange(false);
      }
    }
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [open, onOpenChange, anchorRef]);

  // Move focus into the menu once it is placed, so keyboard users land on the
  // first action instead of being stranded on the trigger.
  useEffect(() => {
    if (open && menuStyle) itemRefs.current[0]?.focus();
  }, [open, menuStyle]);

  function onKeyDown(event: React.KeyboardEvent) {
    const focusable = itemRefs.current.filter((el): el is HTMLButtonElement => el !== null);
    if (focusable.length === 0) return;
    const index = focusable.indexOf(document.activeElement as HTMLButtonElement);
    const step = (delta: number) => {
      event.preventDefault();
      // `indexOf` returns -1 when focus is on the menu container rather than an
      // item (a click on its padding, or the auto-focus effect being pre-empted).
      // Enter from the matching end instead of letting the modulo pick index -2.
      const next =
        index < 0
          ? delta > 0
            ? 0
            : focusable.length - 1
          : (index + delta + focusable.length) % focusable.length;
      focusable[next]?.focus();
    };

    switch (event.key) {
      case 'Escape':
        event.preventDefault();
        close(true);
        break;
      case 'ArrowDown':
        step(1);
        break;
      case 'ArrowUp':
        step(-1);
        break;
      case 'Home':
        event.preventDefault();
        focusable[0]?.focus();
        break;
      case 'End':
        event.preventDefault();
        focusable[focusable.length - 1]?.focus();
        break;
      case 'Tab':
        // A menu is a focus trap in spirit; tabbing out should dismiss it
        // rather than leave an orphaned popover floating over the page.
        // Focus has to go back to the trigger *before* the browser resolves
        // the Tab: the menu is portalled to the end of <body>, so unmounting
        // it first would drop focus to <body> and restart tab order at the
        // top of the app instead of continuing past the sidebar.
        close(true);
        break;
    }
  }

  if (!open || !menuStyle) return null;

  // Drop refs for rows that no longer exist, so arrow/Home/End navigation can
  // never focus a detached button left over from a longer previous `items`.
  itemRefs.current.length = items.length;

  return createPortal(
    <div
      ref={menuRef}
      id={menuId}
      role="menu"
      aria-label={ariaLabel}
      style={menuStyle}
      onKeyDown={onKeyDown}
      className="animate-page-in flex flex-col overflow-y-auto rounded-3xl border border-surface-border-strong bg-surface-elevated p-1.5 shadow-overlay"
    >
      {items.map((item, i) => (
        <ActionMenuRow
          key={item.id}
          item={item}
          itemRef={(el) => {
            itemRefs.current[i] = el;
          }}
          onActivate={() => {
            onOpenChange(false);
            item.onSelect();
          }}
        />
      ))}
    </div>,
    document.body,
  );
}
