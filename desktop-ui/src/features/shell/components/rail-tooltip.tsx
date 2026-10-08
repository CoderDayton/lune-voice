// SPDX-License-Identifier: GPL-3.0-or-later

import { type ReactNode, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { isSidebarExpanded } from '../config/drawer';

const SHOW_DELAY_MS = 300;
const GAP = 8;

/**
 * Names a collapsed-rail control beside it, with its shortcut. Portalled to
 * `<body>` because the sidebar clips overflow, which would cut off a CSS
 * tooltip. Does nothing while the sidebar is expanded: the label is on screen.
 *
 * Wrap exactly one element; the tooltip anchors to it. Listeners are attached
 * to the wrapper directly rather than as JSX props, since the wrapper is not
 * itself interactive — it only observes its child.
 */
export function RailTooltip({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [anchor, setAnchor] = useState<DOMRect | null>(null);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    let timer: number | undefined;

    function show(delay: number) {
      window.clearTimeout(timer);
      if (isSidebarExpanded()) {
        setAnchor(null);
        return;
      }
      timer = window.setTimeout(() => {
        const target = wrapper?.firstElementChild;
        if (target) setAnchor(target.getBoundingClientRect());
      }, delay);
    }
    function hide() {
      window.clearTimeout(timer);
      setAnchor(null);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') hide();
    }
    // Keeps the bubble beside its control when a scroll container moves it.
    const reposition = () => {
      const target = wrapper.firstElementChild;
      if (target) setAnchor((prev) => (prev ? target.getBoundingClientRect() : null));
    };
    // A pointer that leaves must not hide the bubble of a control that still
    // has keyboard focus.
    const onPointerLeave = () => {
      if (!wrapper.contains(document.activeElement)) hide();
    };
    const onPointerEnter = () => show(SHOW_DELAY_MS);
    const onFocusIn = () => show(0);

    wrapper.addEventListener('pointerenter', onPointerEnter);
    wrapper.addEventListener('pointerleave', onPointerLeave);
    window.addEventListener('scroll', reposition, true);
    wrapper.addEventListener('focusin', onFocusIn);
    wrapper.addEventListener('focusout', hide);
    wrapper.addEventListener('keydown', onKeyDown);
    return () => {
      window.clearTimeout(timer);
      wrapper.removeEventListener('pointerenter', onPointerEnter);
      wrapper.removeEventListener('pointerleave', onPointerLeave);
      window.removeEventListener('scroll', reposition, true);
      wrapper.removeEventListener('focusin', onFocusIn);
      wrapper.removeEventListener('focusout', hide);
      wrapper.removeEventListener('keydown', onKeyDown);
    };
  }, []);

  return (
    <div ref={wrapperRef}>
      {children}
      {anchor
        ? createPortal(
            <div
              aria-hidden="true"
              style={{
                position: 'fixed',
                left: anchor.right + GAP,
                top: anchor.top + anchor.height / 2,
                transform: 'translateY(-50%)',
              }}
              className="pointer-events-none z-tooltip flex items-center gap-2.5 rounded-field border border-surface-border bg-surface-elevated px-2.5 py-1.5 text-sm text-content shadow-card"
            >
              <span className="whitespace-nowrap">{label}</span>
              {hint ? (
                <span className="font-mono text-xs whitespace-nowrap text-content-subtle">
                  {hint}
                </span>
              ) : null}
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}
