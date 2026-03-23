
import ReactDOM from "react-dom/client";
import { I18nProvider } from "@lingui/react";
import { i18n } from "@lingui/core";
import { activateLocale, defaultLocale } from "./i18n";
import { DictationPill } from "./features/dictation/dictation-pill";
import "./app.css";


function PillApp() {
  return (
    <I18nProvider i18n={i18n}>
      <DictationPill />
    </I18nProvider>
  );
}

activateLocale(defaultLocale).catch(() => {}).then(() => {
  ReactDOM.createRoot(document.getElementById("pill-root")!).render(
    <PillApp />,
  );
});
