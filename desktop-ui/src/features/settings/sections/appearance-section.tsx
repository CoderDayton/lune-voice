import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import { SettingsRow, SettingsSection } from "../components/section";
import { ThemePicker } from "../components/theme-picker";
import { useLingui } from "@lingui/react/macro";
import {
  getThemePreference,
  isDarkTheme,
  setThemePreference,
  themeLabel,
  watchSystemTheme,
} from "../../../lib/theme";

export function AppearanceSection() {
  const { t } = useLingui();
  const [theme, setTheme] = useState(getThemePreference);

  useEffect(() => {
    return watchSystemTheme(() => setTheme(getThemePreference()));
  }, []);

  const currentLabel = theme === "system" ? t`System` : themeLabel(theme);
  const CurrentIcon = theme === "system" ? Monitor : isDarkTheme(theme) ? Moon : Sun;

  // Quick toggle: lune dark ↔ lune light
  function toggleQuick() {
    const next = isDarkTheme(theme) || theme === "system" ? "lune-light" : "lune";
    setThemePreference(next);
    setTheme(next);
  }

  return (
    <SettingsSection
      title={t`Appearance`}
      description={t`Choose how Lune Voice looks.`}
      icon={Sun}
    >
      <SettingsRow
        label={t`Theme`}
        description={currentLabel}
        control={
          <div className="flex items-center gap-1.5 sm:gap-2">
            <button
              type="button"
              onClick={toggleQuick}
              className="btn btn-sm btn-ghost btn-circle"
              aria-label={t`Toggle dark/light`}
            >
              <CurrentIcon size={16} strokeWidth={1.8} />
            </button>
            <ThemePicker />
          </div>
        }
      />
    </SettingsSection>
  );
}
