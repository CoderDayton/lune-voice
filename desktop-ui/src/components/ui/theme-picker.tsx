import { useLingui } from '@lingui/react/macro';
import { Check, Palette } from 'lucide-react';
import { memo, useCallback, useEffect, useId, useRef, useState } from 'react';
import { setThemePreference, THEME_GROUPS, themeLabel, useThemePreference } from '../../lib/theme';

/**
 * A small app-window mockup rendered inside a `data-theme` scope, so every
 * surface uses the theme's own DaisyUI tokens — no hardcoded colors, and the
 * preview can never drift from the real theme.
 */
function MiniWindow() {
  return (
    <div className="overflow-hidden rounded-md border border-base-300">
      <div className="flex items-center gap-1 border-b border-base-300 bg-base-200 px-1.5 py-1">
        <span className="size-1.5 rounded-full bg-primary" />
        <span className="size-1.5 rounded-full bg-secondary" />
        <span className="size-1.5 rounded-full bg-accent" />
      </div>
      <div className="flex gap-1.5 bg-base-100 p-1.5">
        <div className="w-1/4 shrink-0 rounded-sm bg-neutral" />
        <div className="flex flex-1 flex-col gap-1 py-0.5">
          <div className="h-1.5 w-2/3 rounded-full bg-primary" />
          <div className="h-1.5 w-full rounded-full bg-base-content/30" />
          <div className="h-1.5 w-4/5 rounded-full bg-base-content/15" />
        </div>
      </div>
    </div>
  );
}

/** One light/dark half of the System swatch preview (wrapped in its own theme). */
function SystemHalf() {
  return (
    <div className="flex flex-col gap-1 bg-base-100 p-2">
      <div className="h-1.5 w-3/4 rounded-full bg-primary" />
      <div className="h-1.5 w-full rounded-full bg-accent" />
      <div className="h-1.5 w-2/3 rounded-full bg-base-content/25" />
    </div>
  );
}

/* `content-visibility:auto` lets the browser skip style/layout/paint for the
   swatches scrolled out of view — critical here because each swatch is its own
   `data-theme` scope, so every visible one forces a full DaisyUI variable
   recompute. `contain-intrinsic-size` reserves the height so the scrollbar
   stays stable. The transition is scoped to the properties that actually
   animate (hover lift + selection ring), not `all`. */
const SWATCH_BASE =
  'group relative flex flex-col gap-2 rounded-field border-2 bg-base-100 p-2 text-left text-base-content transition-[transform,box-shadow,border-color] duration-fast [content-visibility:auto] [contain-intrinsic-size:auto_7.5rem] hover:-translate-y-0.5 hover:shadow-card focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary sm:p-2.5';

function selectionClasses(active: boolean) {
  return active ? 'border-primary ring-2 ring-primary/30' : 'border-base-300';
}

const ThemeSwatch = memo(function ThemeSwatch({
  theme,
  active,
  onSelect,
}: {
  theme: string;
  active: boolean;
  onSelect: (theme: string) => void;
}) {
  return (
    <button
      type="button"
      data-theme={theme}
      onClick={() => onSelect(theme)}
      className={`${SWATCH_BASE} ${selectionClasses(active)}`}
    >
      {active && (
        <div className="absolute -top-1.5 -right-1.5 z-10 flex size-5 items-center justify-center rounded-full bg-primary text-primary-content shadow-sm">
          <Check size={12} strokeWidth={3} />
        </div>
      )}
      <span className="truncate text-sm font-medium">{themeLabel(theme)}</span>
      <MiniWindow />
    </button>
  );
});

/**
 * The theme grid on its own, opened by whoever owns the state. Split out from
 * `ThemePicker` so the sidebar's Quick actions menu can raise it without
 * rendering a "Browse themes" button it has no room for.
 */
export function ThemePickerDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useLingui();
  const dialogId = useId();
  const dialogRef = useRef<HTMLDialogElement>(null);
  // This dialog is mounted in two places (the sidebar and Settings), so the
  // selected swatch has to follow the shared preference rather than a copy
  // seeded once at mount.
  const current = useThemePreference();

  // Gate the heavy grid on `open`: 37 themes × a `data-theme`-scoped preview is a
  // lot of style recalc. Mounting it only while the modal is open keeps it out of
  // the DOM on page load and on the other (kept-mounted) settings tabs.
  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (open && !dlg.open) dlg.showModal();
    if (!open && dlg.open) dlg.close();
  }, [open]);

  const handleSelect = useCallback((theme: string) => {
    setThemePreference(theme);
  }, []);

  return (
    <dialog ref={dialogRef} className="modal" aria-labelledby={dialogId} onClose={onClose}>
      <div className="modal-box max-w-2xl sm:max-w-3xl p-0 bg-surface-glass text-content">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-surface-border bg-surface-glass px-4 py-3 sm:px-6 sm:py-4">
          <h3
            id={dialogId}
            className="text-base sm:text-lg font-bold text-content-strong"
          >{t`Choose a theme`}</h3>
          <form method="dialog">
            <button type="submit" className="btn btn-soft btn-sm btn-circle">
              ✕
            </button>
          </form>
        </div>

        <div className="overflow-y-auto max-h-[70vh] px-4 py-4 sm:px-6 sm:py-5 space-y-5 sm:space-y-6">
          {open && (
            <>
              {/* Custom */}
              <div>
                <p className="section-label mb-2 sm:mb-3">{t`Lune`}</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 sm:gap-4">
                  {THEME_GROUPS.custom.map((theme) => (
                    <ThemeSwatch
                      key={theme}
                      theme={theme}
                      active={current === theme}
                      onSelect={handleSelect}
                    />
                  ))}
                  <button
                    type="button"
                    data-theme="lune"
                    onClick={() => handleSelect('system')}
                    className={`${SWATCH_BASE} ${selectionClasses(current === 'system')}`}
                  >
                    {current === 'system' && (
                      <div className="absolute -top-1.5 -right-1.5 z-10 flex size-5 items-center justify-center rounded-full bg-primary text-primary-content shadow-sm">
                        <Check size={12} strokeWidth={3} />
                      </div>
                    )}
                    <span className="truncate text-sm font-medium">{t`System`}</span>
                    <div className="grid grid-cols-2 overflow-hidden rounded-md border border-base-300">
                      <div data-theme="lune">
                        <SystemHalf />
                      </div>
                      <div data-theme="lune-light">
                        <SystemHalf />
                      </div>
                    </div>
                  </button>
                </div>
              </div>

              {/* Light themes */}
              <div>
                <p className="section-label mb-2 sm:mb-3">{t`Light`}</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
                  {THEME_GROUPS.light.map((theme) => (
                    <ThemeSwatch
                      key={theme}
                      theme={theme}
                      active={current === theme}
                      onSelect={handleSelect}
                    />
                  ))}
                </div>
              </div>

              {/* Dark themes */}
              <div>
                <p className="section-label mb-2 sm:mb-3">{t`Dark`}</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
                  {THEME_GROUPS.dark.map((theme) => (
                    <ThemeSwatch
                      key={theme}
                      theme={theme}
                      active={current === theme}
                      onSelect={handleSelect}
                    />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button type="submit">close</button>
      </form>
    </dialog>
  );
}

/** Settings' entry point: the trigger button plus the dialog it raises. */
export function ThemePicker() {
  const { t } = useLingui();
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        className="btn btn-sm btn-soft btn-primary gap-1.5"
        aria-haspopup="dialog"
        onClick={() => setOpen(true)}
      >
        <Palette size={14} strokeWidth={1.8} />
        {t`Browse themes`}
      </button>
      <ThemePickerDialog open={open} onClose={() => setOpen(false)} />
    </>
  );
}
