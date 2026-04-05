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
          <div className="join join-vertical sm:join-horizontal">
            {themeOptions.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                type="button"
                onClick={() => handleChange(value)}
                className={`join-item btn btn-sm gap-1.5 ${
                  theme === value ? "btn-active btn-primary" : "btn-ghost"
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
