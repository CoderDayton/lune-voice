// SPDX-License-Identifier: GPL-3.0-or-later

/** Colour mode: Lune Dark, Lune Light, or whichever the OS prefers. */

import { useSyncExternalStore } from 'react';

export type ThemeMode = 'system' | 'dark' | 'light';
export type ThemeScheme = 'dark' | 'light';

const STORAGE_KEY = 'lune-theme';

/** Stock themes an older build could have saved. They map to their light side;
 *  anything else unrecognised maps to dark. */
const LEGACY_LIGHT_THEMES: ReadonlySet<string> = new Set([
  'lune-light',
  'light',
  'cupcake',
  'bumblebee',
  'emerald',
  'corporate',
  'retro',
  'valentine',
  'garden',
  'lofi',
  'pastel',
  'fantasy',
  'wireframe',
  'cmyk',
  'autumn',
  'acid',
  'lemonade',
  'winter',
  'nord',
  'caramellatte',
  'silk',
]);

/** Reads a saved value, including the theme names older builds stored. */
export function normalizeThemeMode(stored: string | null): ThemeMode {
  if (stored === 'system' || stored === 'dark' || stored === 'light') return stored;
  if (stored && LEGACY_LIGHT_THEMES.has(stored)) return 'light';
  return 'dark';
}

export function themeSchemeFor(mode: ThemeMode, systemPrefersLight: boolean): ThemeScheme {
  if (mode === 'system') return systemPrefersLight ? 'light' : 'dark';
  return mode;
}

/** The explicit mode that changes what the user sees: the opposite of the
 *  look `mode` shows now, so a toggle always has a visible effect. */
export function toggledThemeMode(mode: ThemeMode, systemPrefersLight: boolean): ThemeScheme {
  return themeSchemeFor(mode, systemPrefersLight) === 'light' ? 'dark' : 'light';
}

/* Storage can throw (blocked site data, quota); the theme must still apply,
   so the last mode saved this session backs the read. */
let sessionMode: ThemeMode | null = null;

export function getThemeMode(): ThemeMode {
  try {
    return normalizeThemeMode(localStorage.getItem(STORAGE_KEY));
  } catch {
    return sessionMode ?? 'dark';
  }
}

function saveThemeMode(mode: ThemeMode) {
  sessionMode = mode;
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // Applied for this session only.
  }
}

export function setThemeMode(mode: ThemeMode) {
  saveThemeMode(mode);
  applyTheme(mode);
  for (const listener of listeners) listener();
}

/* The mode lives in localStorage, which fires no event for same-document
   writes. Settings' Appearance and the sidebar's More menu both set it, so
   they need one shared notification or their copies drift apart. */
const listeners = new Set<() => void>();

function subscribeTheme(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** The current mode, re-read whenever any surface changes it. */
export function useThemeMode(): ThemeMode {
  return useSyncExternalStore(subscribeTheme, getThemeMode);
}

/** Saves the normalised mode and applies it. Runs before first paint. */
export function initTheme() {
  const mode = getThemeMode();
  saveThemeMode(mode);
  applyTheme(mode);
}

export function applyTheme(mode: ThemeMode) {
  const scheme = themeSchemeFor(mode, window.matchMedia('(prefers-color-scheme: light)').matches);
  const root = document.documentElement;
  root.setAttribute('data-theme', scheme === 'light' ? 'lune-light' : 'lune');
  root.setAttribute('data-color-scheme', scheme);
}

export function watchSystemTheme(onUpdate?: () => void): () => void {
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const handler = () => {
    if (getThemeMode() === 'system') {
      applyTheme('system');
      for (const listener of listeners) listener();
      onUpdate?.();
    }
  };
  mq.addEventListener('change', handler);
  return () => mq.removeEventListener('change', handler);
}
