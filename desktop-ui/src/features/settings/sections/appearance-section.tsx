import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import { SettingsRow, SettingsSection } from "../components/section";
import { useLingui } from "@lingui/react/macro";
import {
  type ThemePreference,
  getThemePreference,
  setThemePreference,
  watchSystemTheme,
} from "../../../lib/theme";

const themeOptions: { value: ThemePreference; label: string; icon: typeof Sun }[] = [
  { value: "dark", label: "Dark", icon: Moon },
  { value: "light", label: "Light", icon: Sun },
  { value: "system", label: "System", icon: Monitor },
];

export function AppearanceSection() {
  const { t } = useLingui();
  const [theme, setTheme] = useState<ThemePreference>(getThemePreference);

  useEffect(() => {
    return watchSystemTheme(() => setTheme(getThemePreference()));
  }, []);

  function handleChange(value: string) {
    const pref = value as ThemePreference;
    setThemePreference(pref);
    setTheme(pref);
  }

  return (
    <SettingsSection
      title={t`Appearance`}
      description={t`Choose how Lune Voice looks.`}
      icon={Sun}
    >
      <SettingsRow
        label={t`Theme`}
        description={t`Dark and light modes. System follows your OS preference.`}
        control={
          <div className="flex gap-1 rounded-xl border border-surface-border bg-surface-inset p-1">
            {themeOptions.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                type="button"
                onClick={() => handleChange(value)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-colors ${
                  theme === value
                    ? "bg-surface-elevated text-content-strong shadow-sm"
                    : "text-content-subtle hover:text-content-muted"
                }`}
              >
                <Icon size={14} strokeWidth={theme === value ? 2 : 1.5} />
                {label}
              </button>
            ))}
          </div>
        }
      />
    </SettingsSection>
  );
}
