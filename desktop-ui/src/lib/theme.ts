/** Theme preference: dark, light, or follow system. */
export type ThemePreference = "dark" | "light" | "system";

const STORAGE_KEY = "lune-theme";
const DARK_THEME = "lune";
const LIGHT_THEME = "lune-light";

/** Read stored preference or default to "system". */
export function getThemePreference(): ThemePreference {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "dark" || stored === "light" || stored === "system") return stored;
  return "system";
}

/** Save preference and apply the theme. */
export function setThemePreference(pref: ThemePreference) {
  localStorage.setItem(STORAGE_KEY, pref);
  applyTheme(pref);
}

/** Apply theme to the document based on preference. */
export function applyTheme(pref: ThemePreference) {
  const resolved = pref === "system" ? getSystemTheme() : pref;
  document.documentElement.setAttribute("data-theme", resolved === "light" ? LIGHT_THEME : DARK_THEME);
}

/** Detect OS color scheme preference. */
function getSystemTheme(): "dark" | "light" {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/** Listen for OS theme changes when preference is "system". */
export function watchSystemTheme(onUpdate: () => void): () => void {
  const mq = window.matchMedia("(prefers-color-scheme: light)");
  const handler = () => {
    if (getThemePreference() === "system") {
      applyTheme("system");
      onUpdate();
    }
  };
  mq.addEventListener("change", handler);
  return () => mq.removeEventListener("change", handler);
}
