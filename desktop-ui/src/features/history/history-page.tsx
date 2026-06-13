import { useStore } from '@tanstack/react-store';
import { useLayoutEffect, useRef, useState } from 'react';
import { workspaceStore } from '../../store/workspace-store';
import { HistoryTable } from './components/history-table';

/**
 * Sessions page. A single table card that fills the available viewport so the
 * page never scrolls — the table body scrolls internally, with search and
 * pagination pinned. The fill height is measured from the scroll container so
 * it stays correct regardless of the page header's height.
 */
export function HistoryPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  const containerRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number>();

  useLayoutEffect(() => {
    if (!loaded) return;
    const measure = () => {
      const el = containerRef.current;
      if (!el) return;
      const scroller = el.closest('.app-scroll');
      const top = el.getBoundingClientRect().top;
      const bottom = scroller
        ? scroller.getBoundingClientRect().bottom -
          Number.parseFloat(getComputedStyle(scroller).paddingBottom || '0')
        : window.innerHeight - 24;
      setHeight(Math.max(280, Math.floor(bottom - top)));
    };

    measure();
    // Re-measure after the page-in animation settles (it nudges translateY).
    const settle = window.setTimeout(measure, 320);
    window.addEventListener('resize', measure);
    return () => {
      window.clearTimeout(settle);
      window.removeEventListener('resize', measure);
    };
  }, [loaded]);

  if (!loaded) {
    return (
      <div className="panel-surface flex flex-col gap-3 p-4 sm:p-5">
        <div className="skeleton-shimmer h-9 w-full rounded-field" />
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="skeleton-shimmer h-10 w-full rounded-field" />
        ))}
      </div>
    );
  }

  return (
    <div ref={containerRef} style={height ? { height } : undefined} className="overflow-hidden">
      <HistoryTable />
    </div>
  );
}
