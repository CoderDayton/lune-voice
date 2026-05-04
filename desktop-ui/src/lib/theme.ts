/** Theme management — supports custom Lune themes + all 35 DaisyUI built-in themes. */

export type ThemePreference = 'system' | string; // "system" or any theme name

const STORAGE_KEY = 'lune-theme';

/** All available themes grouped by tone. */
export const THEME_GROUPS = {
  custom: ['lune', 'lune-light'],
  light: [
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
  ],
  dark: [
    'dark',
    'synthwave',
    'cyberpunk',
    'halloween',
    'forest',
    'aqua',
    'black',
    'luxury',
    'dracula',
    'business',
    'night',
    'coffee',
    'dim',
    'sunset',
    'abyss',
  ],
} as const;

export const ALL_THEMES = [...THEME_GROUPS.custom, ...THEME_GROUPS.light, ...THEME_GROUPS.dark];

/** Human-readable display name. */
export function themeLabel(id: string): string {
  if (id === 'lune') return 'Lune Dark';
  if (id === 'lune-light') return 'Lune Light';
  return id.charAt(0).toUpperCase() + id.slice(1).replace(/([a-z])([A-Z])/g, '$1 $2');
}

/** Whether a theme is dark-scheme. */
export function isDarkTheme(id: string): boolean {
  if (id === 'lune') return true;
  if (id === 'lune-light') return false;
  return (THEME_GROUPS.dark as readonly string[]).includes(id);
}

export function getThemePreference(): ThemePreference {
  return localStorage.getItem(STORAGE_KEY) ?? 'lune';
}

export function setThemePreference(pref: ThemePreference) {
  localStorage.setItem(STORAGE_KEY, pref);
  applyTheme(pref);
}

export function applyTheme(pref: ThemePreference) {
  const resolved = pref === 'system' ? getSystemTheme() : pref;
  document.documentElement.setAttribute('data-theme', resolved);
  // Propagate light/dark scheme so CSS design tokens auto-switch for all themes
  document.documentElement.setAttribute(
    'data-color-scheme',
    isDarkTheme(resolved) ? 'dark' : 'light',
  );
}

function getSystemTheme(): string {
  if (window.matchMedia('(prefers-color-scheme: light)').matches) return 'lune-light';
  return 'lune';
}

export function watchSystemTheme(onUpdate: () => void): () => void {
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const handler = () => {
    if (getThemePreference() === 'system') {
      applyTheme('system');
      onUpdate();
    }
  };
  mq.addEventListener('change', handler);
  return () => mq.removeEventListener('change', handler);
}
