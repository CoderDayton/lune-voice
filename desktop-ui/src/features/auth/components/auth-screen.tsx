import { useStore } from "@tanstack/react-store";
import { SelectField } from "../../../components/ui/field";
import { authStore } from "../../../store/auth-store";
import { activateLocale, locales, type LocaleKey } from "../../../i18n";
import { saveWorkspaceSetting, workspaceStore } from "../../../store/workspace-store";
import { LuneLogo } from "../../shell/config/icons";
import { Trans } from "@lingui/react/macro";
import { AuthForm } from "./auth-form";
import { AuthHeroPanel } from "./auth-hero-panel";
import { AuthModeSwitch } from "./auth-mode-switch";

export function AuthScreen() {
  const mode = useStore(authStore, (state) => state.mode);
  const language = useStore(workspaceStore, (state) => state.settings.language);

  async function handleLanguageChange(locale: string) {
    await activateLocale(locale);
    void saveWorkspaceSetting("language", locale);
  }

  return (
    <div className="auth-root">
      <div className="auth-backdrop" />
      <div className="auth-shell">
        <section className="card bg-base-200/80 border border-base-300/40 p-4 sm:p-6 flex min-h-0 lg:min-h-[38rem] flex-col">
          <div className="auth-panel-head">
            <div className="auth-brand-row">
              <div className="brand-mark">
                <LuneLogo className="text-white" />
              </div>
              <div className="stack-copy-tight">
                <p className="app-kicker"><Trans>Private desktop dictation</Trans></p>
                <h1 className="brand-title"><Trans>Lune Voice</Trans></h1>
              </div>
            </div>

            <label className="w-full max-w-full sm:max-w-40">
              <span className="sr-only">Language</span>
              <SelectField
                value={language}
                onChange={(event) => void handleLanguageChange(event.target.value)}
              >
                {(Object.entries(locales) as [LocaleKey, string][]).map(([key, label]) => (
                  <option key={key} value={key}>
                    {label}
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
