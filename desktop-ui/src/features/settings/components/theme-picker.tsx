import { useLingui } from '@lingui/react/macro';
import { Check, Palette } from 'lucide-react';
import { useId, useState } from 'react';
import {
  getThemePreference,
  setThemePreference,
  THEME_GROUPS,
  themeLabel,
} from '../../../lib/theme';

const SWATCH_COLORS = ['primary', 'secondary', 'accent', 'neutral'] as const;

const THEME_PREVIEWS: Record<
  string,
  { bg: string; text: string; border: string; colors: string[] }
> = {
  lune: {
    bg: 'oklch(0.145 0.013 238)',
    text: 'oklch(0.93 0.007 238)',
    border: 'oklch(0.90 0.008 238 / 0.20)',
    colors: [
      'oklch(0.74 0.018 245)',
      'oklch(0.62 0.025 250)',
      'oklch(0.68 0.035 70)',
      'oklch(0.24 0.014 238)',
    ],
  },
  'lune-light': {
    bg: 'oklch(0.985 0.006 225)',
    text: 'oklch(0.20 0.016 235)',
    border: 'oklch(0.20 0.010 265 / 0.16)',
    colors: [
      'oklch(0.42 0.020 245)',
      'oklch(0.50 0.024 250)',
      'oklch(0.50 0.040 70)',
      'oklch(0.88 0.008 225)',
    ],
  },
  light: {
    bg: '#ffffff',
    text: '#1f2937',
    border: '#d1d5db',
    colors: ['#570df8', '#f000b8', '#37cdbe', '#3d4451'],
  },
  cupcake: {
    bg: '#faf7f5',
    text: '#291334',
    border: '#e7d8d3',
    colors: ['#65c3c8', '#ef9fbc', '#eeaf3a', '#291334'],
  },
  bumblebee: {
    bg: '#ffffff',
    text: '#181830',
    border: '#e5e7eb',
    colors: ['#e0a82e', '#f9d72f', '#181830', '#181830'],
  },
  emerald: {
    bg: '#ffffff',
    text: '#333c4d',
    border: '#d7dde5',
    colors: ['#66cc8a', '#377cfb', '#ea5234', '#333c4d'],
  },
  corporate: {
    bg: '#ffffff',
    text: '#181a2a',
    border: '#d5d9e2',
    colors: ['#4b6bfb', '#7b92b2', '#67cba0', '#181a2a'],
  },
  retro: {
    bg: '#ece3ca',
    text: '#282425',
    border: '#c6b98f',
    colors: ['#ef9995', '#a4cbb4', '#dc8850', '#2e282a'],
  },
  valentine: {
    bg: '#f0d6e8',
    text: '#632c3b',
    border: '#e6b6d5',
    colors: ['#e96d7b', '#a991f7', '#88dbdd', '#af4670'],
  },
  garden: {
    bg: '#e9e7e7',
    text: '#100f0f',
    border: '#cfcaca',
    colors: ['#5c7f67', '#ecf4e7', '#fae5e5', '#5d5656'],
  },
  lofi: {
    bg: '#ffffff',
    text: '#000000',
    border: '#d5d5d5',
    colors: ['#0d0d0d', '#1a1919', '#262626', '#000000'],
  },
  pastel: {
    bg: '#ffffff',
    text: '#1f2937',
    border: '#e5e7eb',
    colors: ['#d1c1d7', '#f6cbd1', '#b4e9d6', '#70acc7'],
  },
  fantasy: {
    bg: '#ffffff',
    text: '#1f2937',
    border: '#d2d6de',
    colors: ['#6e0b75', '#007ebd', '#f8860d', '#1f2937'],
  },
  wireframe: {
    bg: '#ffffff',
    text: '#1f2937',
    border: '#c8c8c8',
    colors: ['#b8b8b8', '#b8b8b8', '#b8b8b8', '#1f2937'],
  },
  cmyk: {
    bg: '#ffffff',
    text: '#171717',
    border: '#d8d8d8',
    colors: ['#45aeee', '#e8488a', '#fff232', '#1a1a1a'],
  },
  autumn: {
    bg: '#f1f1f1',
    text: '#1f2937',
    border: '#cfcfcf',
    colors: ['#8c0327', '#d85251', '#f28c18', '#2f2f2f'],
  },
  acid: {
    bg: '#fbfbfb',
    text: '#191a3f',
    border: '#d8d8d8',
    colors: ['#ff00f4', '#ff7400', '#cbff00', '#191a3f'],
  },
  lemonade: {
    bg: '#ffffff',
    text: '#1f2937',
    border: '#d9ddc4',
    colors: ['#519903', '#e9e92e', '#f7f9ca', '#1f2937'],
  },
  winter: {
    bg: '#ffffff',
    text: '#394e6a',
    border: '#d7dce4',
    colors: ['#047aff', '#463aa1', '#c148ac', '#394e6a'],
  },
  nord: {
    bg: '#eceff4',
    text: '#2e3440',
    border: '#c9d0dc',
    colors: ['#5e81ac', '#81a1c1', '#88c0d0', '#4c566a'],
  },
  caramellatte: {
    bg: '#fff8f0',
    text: '#3d2b1f',
    border: '#e5c7a3',
    colors: ['#d2691e', '#8b4513', '#f4a460', '#5c4033'],
  },
  silk: {
    bg: '#ffffff',
    text: '#1f2937',
    border: '#d8d5cf',
    colors: ['#343232', '#8a7f8d', '#d6c3a5', '#1f2937'],
  },
  dark: {
    bg: '#1d232a',
    text: '#f8f8f2',
    border: '#3a424d',
    colors: ['#661ae6', '#d926a9', '#1fb2a5', '#191d24'],
  },
  synthwave: {
    bg: '#2d1b69',
    text: '#f9f7fd',
    border: '#5d4aa5',
    colors: ['#e779c1', '#58c7f3', '#f3cc30', '#20134e'],
  },
  cyberpunk: {
    bg: '#ffee00',
    text: '#111111',
    border: '#d7c900',
    colors: ['#ff7598', '#75d1f0', '#c07eec', '#111111'],
  },
  halloween: {
    bg: '#212121',
    text: '#f9f7fd',
    border: '#4a3a2d',
    colors: ['#f28c18', '#6d3a9c', '#51a800', '#1b1d1d'],
  },
  forest: {
    bg: '#171212',
    text: '#e5e7eb',
    border: '#394236',
    colors: ['#1eb854', '#1fd65f', '#d99330', '#19362d'],
  },
  aqua: {
    bg: '#193549',
    text: '#d4f1f4',
    border: '#2d6277',
    colors: ['#09ecf3', '#966fb3', '#ffe999', '#2a323c'],
  },
  black: {
    bg: '#000000',
    text: '#ffffff',
    border: '#333333',
    colors: ['#343232', '#343232', '#343232', '#181818'],
  },
  luxury: {
    bg: '#171618',
    text: '#e2d9c3',
    border: '#4b4134',
    colors: ['#ffffff', '#152747', '#513448', '#2b211e'],
  },
  dracula: {
    bg: '#282a36',
    text: '#f8f8f2',
    border: '#44475a',
    colors: ['#ff79c6', '#bd93f9', '#ffb86c', '#414558'],
  },
  business: {
    bg: '#202020',
    text: '#f5f5f5',
    border: '#3f3f46',
    colors: ['#1c4e80', '#7c909a', '#ea6947', '#1f2937'],
  },
  night: {
    bg: '#0f172a',
    text: '#e5e7eb',
    border: '#25314f',
    colors: ['#38bdf8', '#818cf8', '#f471b5', '#1e293b'],
  },
  coffee: {
    bg: '#20161f',
    text: '#f5e6d3',
    border: '#4b3621',
    colors: ['#db924b', '#263e3f', '#10576d', '#2a1f1d'],
  },
  dim: {
    bg: '#2a303c',
    text: '#d7dde4',
    border: '#465064',
    colors: ['#9fe88d', '#ff7d5c', '#c792ea', '#1c212b'],
  },
  sunset: {
    bg: '#1e1e2e',
    text: '#f5e0dc',
    border: '#45475a',
    colors: ['#f38ba8', '#fab387', '#f9e2af', '#313244'],
  },
  abyss: {
    bg: '#000617',
    text: '#dbeafe',
    border: '#172554',
    colors: ['#2563eb', '#7c3aed', '#06b6d4', '#0f172a'],
  },
};

function getThemePreview(theme: string) {
  return THEME_PREVIEWS[theme] ?? THEME_PREVIEWS.lune;
}

function ThemeSwatch({
  theme,
  active,
  onSelect,
}: {
  theme: string;
  active: boolean;
  onSelect: () => void;
}) {
  const preview = getThemePreview(theme);

  return (
    <button
      type="button"
      onClick={onSelect}
      className="group relative flex flex-col gap-1.5 rounded-field border-2 p-2 text-left transition-all hover:-translate-y-0.5 hover:shadow-card sm:p-3"
      style={{
        background: preview.bg,
        borderColor: active ? 'var(--color-primary)' : preview.border,
        boxShadow: active
          ? '0 0 0 2px color-mix(in oklab, var(--color-primary) 25%, transparent)'
          : undefined,
        color: preview.text,
      }}
    >
      {active && (
        <div className="absolute -top-1.5 -right-1.5 flex size-5 items-center justify-center rounded-full bg-primary text-primary-content">
          <Check size={12} strokeWidth={3} />
        </div>
      )}
      <span className="text-sm font-medium truncate">{themeLabel(theme)}</span>
      <div className="flex gap-1">
        {SWATCH_COLORS.map((color, index) => (
          <div
            key={color}
            className="size-4 rounded-selector border border-black/10 sm:size-5"
            style={{ backgroundColor: preview.colors[index] }}
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

  const systemPreview = getThemePreview('lune');

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
        ref={(el) => {
          setDialog(el);
          return () => setDialog(null);
        }}
        className="modal"
        aria-labelledby={dialogId}
      >
        <div className="modal-box max-w-2xl sm:max-w-3xl p-0 bg-surface-glass text-content">
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-surface-border bg-surface-glass px-4 py-3 backdrop-blur-xl sm:px-6 sm:py-4">
            <h3
              id={dialogId}
              className="text-base sm:text-lg font-bold text-content-strong"
            >{t`Choose a theme`}</h3>
            <form method="dialog">
              <button type="submit" className="btn btn-ghost btn-sm btn-circle">
                ✕
              </button>
            </form>
          </div>

          <div className="overflow-y-auto max-h-[70vh] px-4 py-4 sm:px-6 sm:py-5 space-y-5 sm:space-y-6">
            {/* Custom */}
            <div>
              <p className="section-label mb-2 sm:mb-3">{t`Lune`}</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 sm:gap-4">
                {THEME_GROUPS.custom.map((t) => (
                  <ThemeSwatch
                    key={t}
                    theme={t}
                    active={current === t}
                    onSelect={() => handleSelect(t)}
                  />
                ))}
                <button
                  type="button"
                  onClick={() => handleSelect('system')}
                  className="relative flex flex-col gap-1.5 rounded-field border-2 p-2 text-left transition-all hover:-translate-y-0.5 hover:shadow-card sm:p-3"
                  style={{
                    background:
                      'linear-gradient(135deg, oklch(0.145 0.013 238) 0 50%, oklch(0.985 0.006 225) 50% 100%)',
                    borderColor:
                      current === 'system'
                        ? 'var(--color-primary)'
                        : 'var(--surface-border-strong)',
                    boxShadow:
                      current === 'system'
                        ? '0 0 0 2px color-mix(in oklab, var(--color-primary) 25%, transparent)'
                        : undefined,
                    color: systemPreview.text,
                  }}
                >
                  {current === 'system' && (
                    <div className="absolute -top-1.5 -right-1.5 flex size-5 items-center justify-center rounded-full bg-primary text-primary-content">
                      <Check size={12} strokeWidth={3} />
                    </div>
                  )}
                  <span className="text-sm font-medium">{t`System`}</span>
                  <div className="flex gap-1">
                    {[
                      'oklch(0.74 0.018 245)',
                      'oklch(0.62 0.025 250)',
                      'oklch(0.68 0.035 70)',
                      'oklch(0.88 0.008 225)',
                    ].map((color) => (
                      <div
                        key={color}
                        className="size-4 rounded-selector border border-black/10 sm:size-5"
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </div>
                </button>
              </div>
            </div>

            {/* Light themes */}
            <div>
              <p className="section-label mb-2 sm:mb-3">{t`Light`}</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
                {THEME_GROUPS.light.map((t) => (
                  <ThemeSwatch
                    key={t}
                    theme={t}
                    active={current === t}
                    onSelect={() => handleSelect(t)}
                  />
                ))}
              </div>
            </div>

            {/* Dark themes */}
            <div>
              <p className="section-label mb-2 sm:mb-3">{t`Dark`}</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
                {THEME_GROUPS.dark.map((t) => (
                  <ThemeSwatch
                    key={t}
                    theme={t}
                    active={current === t}
                    onSelect={() => handleSelect(t)}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
        <form method="dialog" className="modal-backdrop">
          <button type="submit">close</button>
        </form>
      </dialog>
    </>
  );
}
