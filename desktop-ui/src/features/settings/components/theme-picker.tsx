import { useId, useState } from "react";
import { Check, Palette } from "lucide-react";
import { useLingui } from "@lingui/react/macro";
import {
  THEME_GROUPS,
  getThemePreference,
  setThemePreference,
  themeLabel,
} from "../../../lib/theme";

const SWATCH_COLORS = ["primary", "secondary", "accent", "neutral"] as const;

function ThemeSwatch({ theme, active, onSelect }: { theme: string; active: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      data-theme={theme}
      onClick={onSelect}
      className={`group relative flex flex-col gap-1.5 rounded-field border-2 text-base-content p-2 sm:p-3 transition-all text-left ${
        active
          ? "border-primary ring-2 ring-primary/20"
          : "border-base-300/50 hover:border-base-content/20"
      }`}
      style={{ background: "var(--color-base-100)" }}
    >
      {active && (
        <div className="absolute -top-1.5 -right-1.5 flex size-5 items-center justify-center rounded-full bg-primary text-primary-content">
          <Check size={12} strokeWidth={3} />
        </div>
      )}
      <span className="text-xs font-medium truncate">
        {themeLabel(theme)}
      </span>
      <div className="flex gap-1">
        {SWATCH_COLORS.map((color) => (
          <div
            key={color}
            className="size-4 sm:size-5 rounded-selector"
            style={{ backgroundColor: `var(--color-${color})` }}
          />
        ))}
      </div>
    </button>
  );
}

export function ThemePicker() {
  const { t } = useLingui();
  const dialogId = useId();
  const [dialog, setDialog] = useState<HTMLDialogElement | null>(null);
  const [current, setCurrent] = useState(getThemePreference);

  function handleSelect(theme: string) {
    setThemePreference(theme);
    setCurrent(theme);
  }

  return (
    <>
      <button
        type="button"
        className="btn btn-sm btn-soft btn-primary gap-1.5"
        aria-haspopup="dialog"
        onClick={() => dialog?.showModal()}
      >
        <Palette size={14} strokeWidth={1.8} />
        {t`Browse themes`}
      </button>

      <dialog
        ref={(el) => { setDialog(el); return () => setDialog(null); }}
        className="modal"
        aria-labelledby={dialogId}
      >
        <div className="modal-box max-w-2xl sm:max-w-3xl p-0">
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-base-300/20 bg-base-100 px-4 py-3 sm:px-6 sm:py-4">
            <h3 id={dialogId} className="text-base sm:text-lg font-bold">{t`Choose a theme`}</h3>
            <form method="dialog">
              <button type="submit" className="btn btn-ghost btn-sm btn-circle">✕</button>
            </form>
          </div>

          <div className="overflow-y-auto max-h-[70vh] px-4 py-4 sm:px-6 sm:py-5 space-y-5 sm:space-y-6">
            {/* Custom */}
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-content-subtle mb-2 sm:mb-3">{t`Lune`}</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 sm:gap-3">
                {THEME_GROUPS.custom.map((t) => (
                  <ThemeSwatch key={t} theme={t} active={current === t} onSelect={() => handleSelect(t)} />
                ))}
                <button
                  type="button"
                  onClick={() => handleSelect("system")}
                  className={`flex flex-col gap-1.5 rounded-field border-2 p-2 sm:p-3 transition-all text-left ${
                    current === "system"
                      ? "border-primary ring-2 ring-primary/20"
                      : "border-base-300/50 hover:border-base-content/20"
                  }`}
                >
                  {current === "system" && (
                    <div className="absolute -top-1.5 -right-1.5 flex size-5 items-center justify-center rounded-full bg-primary text-primary-content">
                      <Check size={12} strokeWidth={3} />
                    </div>
                  )}
                  <span className="text-xs font-medium">{t`System`}</span>
                  <div className="flex gap-1">
                    <div className="size-4 sm:size-5 rounded-selector bg-base-300/50 border border-base-300" />
                    <div className="size-4 sm:size-5 rounded-selector bg-base-300/50 border border-base-300" />
                    <div className="size-4 sm:size-5 rounded-selector bg-base-300/50 border border-base-300" />
                    <div className="size-4 sm:size-5 rounded-selector bg-base-300/50 border border-base-300" />
                  </div>
                </button>
              </div>
            </div>

            {/* Light themes */}
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-content-subtle mb-2 sm:mb-3">{t`Light`}</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 sm:gap-3">
                {THEME_GROUPS.light.map((t) => (
                  <ThemeSwatch key={t} theme={t} active={current === t} onSelect={() => handleSelect(t)} />
                ))}
              </div>
            </div>

            {/* Dark themes */}
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-content-subtle mb-2 sm:mb-3">{t`Dark`}</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 sm:gap-3">
                {THEME_GROUPS.dark.map((t) => (
                  <ThemeSwatch key={t} theme={t} active={current === t} onSelect={() => handleSelect(t)} />
                ))}
              </div>
            </div>
          </div>
        </div>
        <form method="dialog" className="modal-backdrop"><button type="submit">close</button></form>
      </dialog>
    </>
  );
}
