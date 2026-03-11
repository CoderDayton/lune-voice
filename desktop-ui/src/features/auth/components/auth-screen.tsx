import { useStore } from "@tanstack/react-store";
import { useState } from "react";
import { SelectField } from "../../../components/ui/field";
import { authStore } from "../../../store/auth-store";
import { LuneLogo } from "../../shell/config/icons";
import { AuthForm } from "./auth-form";
import { AuthHeroPanel } from "./auth-hero-panel";
import { AuthModeSwitch } from "./auth-mode-switch";

const languageOptions = ["English", "Spanish", "German"];

export function AuthScreen() {
  const [language, setLanguage] = useState(languageOptions[0]!);
  const mode = useStore(authStore, (state) => state.mode);

  return (
    <div className="auth-root">
      <div className="auth-backdrop" />
      <div className="auth-shell">
        <section className="auth-panel">
          <div className="auth-panel-head">
            <div className="auth-brand-row">
              <div className="brand-mark">
                <LuneLogo className="text-white" />
              </div>
              <div className="stack-copy-tight">
                <p className="app-kicker">Private desktop dictation</p>
                <h1 className="brand-title">Lune Voice</h1>
              </div>
            </div>

            <label className="auth-language-picker">
              <span className="sr-only">Language</span>
              <SelectField
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                className="auth-language-select"
              >
                {languageOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </SelectField>
            </label>
          </div>

          <div className="auth-panel-body">
            <AuthModeSwitch />
            <AuthForm key={mode} />
          </div>
        </section>

        <AuthHeroPanel />
      </div>
    </div>
  );
}
