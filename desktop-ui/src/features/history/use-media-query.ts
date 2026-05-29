import { useEffect, useState } from 'react';

/** Subscribes to a CSS media query and returns whether it currently matches.
 *  Used to switch the History view between a stacked (modal) layout on narrow
 *  screens and a master-detail two-pane layout on desktop. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(query).matches,
  );

  useEffect(() => {
    const mq = window.matchMedia(query);
    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    setMatches(mq.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [query]);

  return matches;
}
