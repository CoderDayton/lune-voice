#!/usr/bin/env -S npx tsx
/**
 * OKLCh Palette Generator for DaisyUI 5 Themes
 *
 * Generates mathematically correct dark + light theme palettes using:
 * - Perceptually uniform OKLCh color space
 * - WCAG 2.x contrast ratio validation
 * - Sine-curve chroma distribution (peak at mid-tones)
 * - Material Design 3 surface elevation theory
 * - Harmonic hue relationships (analogous, split-complementary)
 *
 * Usage:
 *   npx tsx scripts/generate-palette.ts                    # default (hue 270, purple)
 *   npx tsx scripts/generate-palette.ts --hue 280          # custom brand hue
 *   npx tsx scripts/generate-palette.ts --hue 200 --css    # output as CSS
 *   npx tsx scripts/generate-palette.ts --hue 280 --json   # output as JSON
 */

// ── OKLCh Math ──────────────────────────────────────────────────────────

interface OklchColor {
  l: number; // 0-1
  c: number; // 0-0.4
  h: number; // 0-360
}

/** Clamp a value to [min, max]. */
const clamp = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v));

/** Normalize hue to [0, 360). */
const normalizeHue = (h: number) => ((h % 360) + 360) % 360;

/** Format an OKLCh color as a CSS oklch() string. */
function fmt(color: OklchColor, alpha?: number): string {
  const l = (color.l * 100).toFixed(1);
  const c = color.c.toFixed(3);
  const h = color.h.toFixed(0);
  if (alpha !== undefined && alpha < 1) {
    return `oklch(${l}% ${c} ${h} / ${alpha.toFixed(2)})`;
  }
  return `oklch(${l}% ${c} ${h})`;
}

// ── Contrast Calculation ────────────────────────────────────────────────

/**
 * Approximate sRGB relative luminance from OKLCh lightness.
 *
 * OKLCh L is perceptual lightness. Relative luminance Y ≈ L^3 is a
 * reasonable approximation for contrast calculations when we only have
 * OKLCh values (exact conversion requires going through OKLab → linear RGB).
 */
function approxLuminance(l: number): number {
  // Attempt a closer mapping: OKLCh L=0 → Y=0, L=1 → Y=1
  // L^3 is a decent approximation of the perceptual→physical relationship
  return l * l * l;
}

/** WCAG 2.x contrast ratio between two OKLCh lightness values. */
function contrastRatio(l1: number, l2: number): number {
  const y1 = approxLuminance(l1);
  const y2 = approxLuminance(l2);
  const lighter = Math.max(y1, y2);
  const darker = Math.min(y1, y2);
  return (lighter + 0.05) / (darker + 0.05);
}

/** Check WCAG AA (4.5:1 for normal text, 3:1 for large text / UI). */
function meetsAA(bgL: number, fgL: number): boolean {
  return contrastRatio(bgL, fgL) >= 4.5;
}

// ── Chroma Curves ───────────────────────────────────────────────────────

/**
 * Sine-curve chroma: peaks at mid-lightness, tapers at extremes.
 * This mimics how saturated colors exist in the middle of the lightness
 * range and desaturate toward black and white.
 */
function sineCurveChroma(l: number, maxChroma: number): number {
  return maxChroma * Math.sin(l * Math.PI);
}

/**
 * Max chroma that stays in sRGB gamut for a given L and H.
 * Conservative estimate — real gamut boundary is complex, so we use
 * empirically safe values.
 */
function safeMaxChroma(l: number, h: number): number {
  // Gamut narrows at extremes of lightness
  const lFactor = Math.sin(l * Math.PI); // 0 at L=0 and L=1
  // Some hues have narrower gamut (yellows ~90° and blues ~270° at high chroma)
  const hRad = (h * Math.PI) / 180;
  const hFactor = 0.85 + 0.15 * Math.abs(Math.sin(hRad)); // slight hue compensation
  return clamp(0.32 * lFactor * hFactor, 0, 0.35);
}

// ── Palette Generation ──────────────────────────────────────────────────

interface ThemePalette {
  name: string;
  colorScheme: "dark" | "light";
  base100: OklchColor;
  base200: OklchColor;
  base300: OklchColor;
  baseContent: OklchColor;
  primary: OklchColor;
  primaryContent: OklchColor;
  secondary: OklchColor;
  secondaryContent: OklchColor;
  accent: OklchColor;
  accentContent: OklchColor;
  neutral: OklchColor;
  neutralContent: OklchColor;
  info: OklchColor;
  infoContent: OklchColor;
  success: OklchColor;
  successContent: OklchColor;
  warning: OklchColor;
  warningContent: OklchColor;
  error: OklchColor;
  errorContent: OklchColor;
  // Surface tokens for the design system
  surfaces: {
    card: { color: OklchColor; alpha: number };
    elevated: { color: OklchColor; alpha: number };
    glass: { color: OklchColor; alpha: number };
    navActive: { color: OklchColor; alpha: number };
    hover: { color: OklchColor; alpha: number };
    border: { color: OklchColor; alpha: number };
    inset: { color: OklchColor; alpha: number };
  };
  // Content tokens (text hierarchy)
  content: {
    strong: OklchColor;
    default: OklchColor;
    muted: OklchColor;
    subtle: OklchColor;
    dim: OklchColor;
  };
  // Body gradient
  bodyGradient: { from: OklchColor; to: OklchColor };
}

/**
 * Generate a complete dark theme palette from a brand hue.
 *
 * Uses Material Design 3 surface elevation theory adapted to OKLCh:
 * - Surfaces step up in lightness by +0.04 per elevation level
 * - Neutral surfaces carry a subtle tint of the brand hue (C ≈ 0.012)
 * - Text uses solid lightness steps (no alpha) for consistent contrast
 * - Semantic colors placed at perceptually optimal hue positions
 */
function generateDarkPalette(brandHue: number): ThemePalette {
  const h = normalizeHue(brandHue);
  const neutralChroma = 0.012; // subtle brand tint on neutrals

  // ── Base surfaces (M3 elevation: lowest → highest) ──
  // L values: 0.15 → 0.19 → 0.24 (wider steps than before for readability)
  const base100: OklchColor = { l: 0.15, c: neutralChroma, h };
  const base200: OklchColor = { l: 0.19, c: neutralChroma, h };
  const base300: OklchColor = { l: 0.24, c: neutralChroma, h };
  const baseContent: OklchColor = { l: 0.93, c: 0.006, h };

  // ── Primary: brand color at readable-on-dark lightness ──
  // L ~0.65, high chroma for vibrancy
  const primaryChroma = clamp(sineCurveChroma(0.65, 0.28), 0, safeMaxChroma(0.65, h));
  const primary: OklchColor = { l: 0.65, c: primaryChroma, h };
  const primaryContent: OklchColor = { l: 0.98, c: 0.005, h };

  // ── Secondary: analogous hue (+40°), slightly cooler ──
  const secHue = normalizeHue(h - 40);
  const secChroma = clamp(sineCurveChroma(0.70, 0.22), 0, safeMaxChroma(0.70, secHue));
  const secondary: OklchColor = { l: 0.70, c: secChroma, h: secHue };
  const secondaryContent: OklchColor = { l: 0.98, c: 0.005, h: secHue };

  // ── Accent: split-complementary (+150°) ──
  const accHue = normalizeHue(h + 150);
  const accChroma = clamp(sineCurveChroma(0.75, 0.22), 0, safeMaxChroma(0.75, accHue));
  const accent: OklchColor = { l: 0.75, c: accChroma, h: accHue };
  const accentContent: OklchColor = { l: 0.18, c: 0.04, h: accHue };

  // ── Neutral: low-chroma brand tint ──
  const neutral: OklchColor = { l: 0.24, c: neutralChroma, h };
  const neutralContent: OklchColor = { l: 0.88, c: 0.008, h };

  // ── Semantic colors at fixed hue positions ──
  const info: OklchColor = { l: 0.68, c: 0.14, h: 235 };
  const infoContent: OklchColor = { l: 0.98, c: 0.005, h: 235 };

  const success: OklchColor = { l: 0.67, c: 0.16, h: 150 };
  const successContent: OklchColor = { l: 0.98, c: 0.005, h: 150 };

  const warning: OklchColor = { l: 0.80, c: 0.15, h: 80 };
  const warningContent: OklchColor = { l: 0.22, c: 0.04, h: 80 };

  const error: OklchColor = { l: 0.62, c: 0.20, h: 25 };
  const errorContent: OklchColor = { l: 0.98, c: 0.005, h: 25 };

  // ── Surface tokens (card elevation hierarchy) ──
  // Cards are +0.06 L above base-100, elevated +0.09 above base-100
  const surfaces = {
    card:      { color: { l: 0.21, c: neutralChroma, h }, alpha: 0.60 },
    elevated:  { color: { l: 0.25, c: neutralChroma, h }, alpha: 0.75 },
    glass:     { color: { l: 0.18, c: neutralChroma, h }, alpha: 0.92 },
    navActive: { color: { ...primary },                    alpha: 0.12 },
    hover:     { color: { l: 1.00, c: 0, h: 0 },         alpha: 0.05 },
    border:    { color: { l: 0.90, c: 0.008, h },         alpha: 0.12 },
    inset:     { color: { l: 0.13, c: neutralChroma, h }, alpha: 0.55 },
  };

  // ── Content hierarchy (solid lightness, no alpha) ──
  // Steps: 0.94, 0.86, 0.68, 0.52, 0.38 — each ≥0.14 apart for clarity
  const content = {
    strong:  { l: 0.94, c: 0.006, h },
    default: { l: 0.86, c: 0.006, h },
    muted:   { l: 0.68, c: 0.008, h },
    subtle:  { l: 0.52, c: 0.008, h },
    dim:     { l: 0.38, c: 0.008, h },
  };

  // ── Body gradient ──
  const bodyGradient = {
    from: { l: 0.16, c: 0.014, h: normalizeHue(h + 2) },
    to:   { l: 0.12, c: 0.010, h: normalizeHue(h - 2) },
  };

  return {
    name: "lune",
    colorScheme: "dark",
    base100, base200, base300, baseContent,
    primary, primaryContent,
    secondary, secondaryContent,
    accent, accentContent,
    neutral, neutralContent,
    info, infoContent,
    success, successContent,
    warning, warningContent,
    error, errorContent,
    surfaces,
    content,
    bodyGradient,
  };
}

/**
 * Generate a complete light theme palette from a brand hue.
 *
 * Light mode inverts the elevation model:
 * - Base surfaces decrease in lightness from 0.98 → 0.91
 * - Primary and semantic colors are darker (L ~0.50-0.55) for contrast
 * - Content tokens descend from dark (L=0.18) to light (L=0.70)
 */
function generateLightPalette(brandHue: number): ThemePalette {
  const h = normalizeHue(brandHue);
  const neutralChroma = 0.006; // very subtle tint in light mode

  const base100: OklchColor = { l: 0.98, c: 0.005, h };
  const base200: OklchColor = { l: 0.95, c: 0.006, h };
  const base300: OklchColor = { l: 0.91, c: 0.008, h };
  const baseContent: OklchColor = { l: 0.22, c: 0.012, h };

  // Primary darker for contrast against light bg
  const primaryChroma = clamp(sineCurveChroma(0.52, 0.30), 0, safeMaxChroma(0.52, h));
  const primary: OklchColor = { l: 0.52, c: primaryChroma, h };
  const primaryContent: OklchColor = { l: 0.98, c: 0.005, h };

  const secHue = normalizeHue(h - 40);
  const secChroma = clamp(sineCurveChroma(0.52, 0.25), 0, safeMaxChroma(0.52, secHue));
  const secondary: OklchColor = { l: 0.52, c: secChroma, h: secHue };
  const secondaryContent: OklchColor = { l: 0.98, c: 0.005, h: secHue };

  const accHue = normalizeHue(h + 150);
  const accChroma = clamp(sineCurveChroma(0.55, 0.25), 0, safeMaxChroma(0.55, accHue));
  const accent: OklchColor = { l: 0.55, c: accChroma, h: accHue };
  const accentContent: OklchColor = { l: 0.98, c: 0.005, h: accHue };

  const neutral: OklchColor = { l: 0.88, c: neutralChroma, h };
  const neutralContent: OklchColor = { l: 0.30, c: 0.012, h };

  const info: OklchColor = { l: 0.52, c: 0.16, h: 235 };
  const infoContent: OklchColor = { l: 0.98, c: 0.005, h: 235 };

  const success: OklchColor = { l: 0.50, c: 0.17, h: 150 };
  const successContent: OklchColor = { l: 0.98, c: 0.005, h: 150 };

  const warning: OklchColor = { l: 0.70, c: 0.16, h: 80 };
  const warningContent: OklchColor = { l: 0.22, c: 0.04, h: 80 };

  const error: OklchColor = { l: 0.52, c: 0.22, h: 25 };
  const errorContent: OklchColor = { l: 0.98, c: 0.005, h: 25 };

  const surfaces = {
    card:      { color: { l: 1.00, c: 0, h: 0 },         alpha: 0.85 },
    elevated:  { color: { l: 1.00, c: 0, h: 0 },         alpha: 0.95 },
    glass:     { color: { l: 0.99, c: 0.003, h },         alpha: 0.95 },
    navActive: { color: { ...primary },                    alpha: 0.10 },
    hover:     { color: { l: 0.00, c: 0, h: 0 },         alpha: 0.04 },
    border:    { color: { l: 0.20, c: 0.010, h },         alpha: 0.12 },
    inset:     { color: { l: 0.94, c: 0.005, h },         alpha: 0.80 },
  };

  const content = {
    strong:  { l: 0.18, c: 0.015, h },
    default: { l: 0.25, c: 0.012, h },
    muted:   { l: 0.42, c: 0.010, h },
    subtle:  { l: 0.58, c: 0.008, h },
    dim:     { l: 0.72, c: 0.006, h },
  };

  const bodyGradient = {
    from: { l: 0.96, c: 0.004, h },
    to:   { l: 0.93, c: 0.006, h: normalizeHue(h - 2) },
  };

  return {
    name: "lune-light",
    colorScheme: "light",
    base100, base200, base300, baseContent,
    primary, primaryContent,
    secondary, secondaryContent,
    accent, accentContent,
    neutral, neutralContent,
    info, infoContent,
    success, successContent,
    warning, warningContent,
    error, errorContent,
    surfaces,
    content,
    bodyGradient,
  };
}

// ── Output Formatters ───────────────────────────────────────────────────

function toDaisyCSS(palette: ThemePalette): string {
  const p = palette;
  return `@plugin "daisyui/theme" {
  name: "${p.name}";
  default: ${p.name === "lune" ? "true" : "false"};
  color-scheme: ${p.colorScheme};
  --color-base-100: ${fmt(p.base100)};
  --color-base-200: ${fmt(p.base200)};
  --color-base-300: ${fmt(p.base300)};
  --color-base-content: ${fmt(p.baseContent)};
  --color-primary: ${fmt(p.primary)};
  --color-primary-content: ${fmt(p.primaryContent)};
  --color-secondary: ${fmt(p.secondary)};
  --color-secondary-content: ${fmt(p.secondaryContent)};
  --color-accent: ${fmt(p.accent)};
  --color-accent-content: ${fmt(p.accentContent)};
  --color-neutral: ${fmt(p.neutral)};
  --color-neutral-content: ${fmt(p.neutralContent)};
  --color-info: ${fmt(p.info)};
  --color-info-content: ${fmt(p.infoContent)};
  --color-success: ${fmt(p.success)};
  --color-success-content: ${fmt(p.successContent)};
  --color-warning: ${fmt(p.warning)};
  --color-warning-content: ${fmt(p.warningContent)};
  --color-error: ${fmt(p.error)};
  --color-error-content: ${fmt(p.errorContent)};
  --radius-selector: 0.75rem;
  --radius-field: 0.75rem;
  --radius-box: 1.25rem;
  --size-selector: 0.25rem;
  --size-field: 0.25rem;
  --border: 1px;
  --depth: 1;
  --noise: 0;
}`;
}

function toSurfaceCSS(palette: ThemePalette): string {
  const s = palette.surfaces;
  const c = palette.content;
  const selector =
    palette.colorScheme === "dark"
      ? ':root, [data-color-scheme="dark"]'
      : '[data-color-scheme="light"]';

  return `/* ${palette.colorScheme === "dark" ? "Dark" : "Light"} scheme tokens */
${selector} {
  --content-strong: ${fmt(c.strong)};
  --content-default: ${fmt(c.default)};
  --content-muted: ${fmt(c.muted)};
  --content-subtle: ${fmt(c.subtle)};
  --content-dim: ${fmt(c.dim)};
  --surface-card: ${fmt(s.card.color, s.card.alpha)};
  --surface-elevated: ${fmt(s.elevated.color, s.elevated.alpha)};
  --surface-glass: ${fmt(s.glass.color, s.glass.alpha)};
  --surface-nav-active: ${fmt(s.navActive.color, s.navActive.alpha)};
  --surface-hover: ${fmt(s.hover.color, s.hover.alpha)};
  --surface-border: ${fmt(s.border.color, s.border.alpha)};
  --surface-inset: ${fmt(s.inset.color, s.inset.alpha)};
}`;
}

function toBodyCSS(palette: ThemePalette): string {
  const g = palette.bodyGradient;
  if (palette.colorScheme === "dark") {
    return `  --surface-body:
    linear-gradient(170deg, ${fmt(g.from)}, ${fmt(g.to)});`;
  }
  return `[data-color-scheme="light"] {
  --surface-body:
    linear-gradient(170deg, ${fmt(g.from)}, ${fmt(g.to)});
}`;
}

function printContrastReport(palette: ThemePalette) {
  const bgL = palette.base100.l;
  const checks = [
    ["content-strong on base-100", palette.content.strong.l, bgL],
    ["content-default on base-100", palette.content.default.l, bgL],
    ["content-muted on base-100", palette.content.muted.l, bgL],
    ["content-subtle on base-100", palette.content.subtle.l, bgL],
    ["primary on base-100", palette.primary.l, bgL],
    ["base-content on base-100", palette.baseContent.l, bgL],
    ["primary-content on primary", palette.primaryContent.l, palette.primary.l],
  ] as const;

  console.log(`\n── Contrast Report (${palette.colorScheme}) ──`);
  console.log("  Target: WCAG AA 4.5:1 for normal text\n");
  for (const [label, fgL, bgLv] of checks) {
    const ratio = contrastRatio(fgL, bgLv);
    const pass = ratio >= 4.5;
    const icon = pass ? "✓" : "✗";
    console.log(
      `  ${icon} ${ratio.toFixed(1)}:1  ${label}  (ΔL=${Math.abs(fgL - bgLv).toFixed(2)})`,
    );
  }
}

// ── CLI ──────────────────────────────────────────────────────────────────

function main() {
  const args = process.argv.slice(2);
  let brandHue = 270; // default: Lune purple
  let outputMode: "pretty" | "css" | "json" = "pretty";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--hue" && args[i + 1]) {
      brandHue = Number.parseFloat(args[++i]);
    } else if (args[i] === "--css") {
      outputMode = "css";
    } else if (args[i] === "--json") {
      outputMode = "json";
    } else if (args[i] === "--help" || args[i] === "-h") {
      console.log("Usage: generate-palette [--hue <0-360>] [--css | --json]");
      console.log("  --hue   Brand hue in degrees (default: 270 purple)");
      console.log("  --css   Output as CSS @plugin blocks for app.css");
      console.log("  --json  Output as JSON");
      process.exit(0);
    }
  }

  const dark = generateDarkPalette(brandHue);
  const light = generateLightPalette(brandHue);

  if (outputMode === "json") {
    console.log(JSON.stringify({ dark, light }, null, 2));
    return;
  }

  if (outputMode === "css") {
    console.log("/* ── Generated DaisyUI themes ── */\n");
    console.log(toDaisyCSS(dark));
    console.log();
    console.log(toDaisyCSS(light));
    console.log();
    console.log("/* ── Design system surface tokens ── */\n");
    console.log(toSurfaceCSS(dark));
    console.log();
    console.log(toSurfaceCSS(light));
    console.log();
    console.log("/* ── Body gradients ── */\n");
    console.log(":root {");
    console.log(toBodyCSS(dark));
    console.log("}");
    console.log();
    console.log(toBodyCSS(light));
    return;
  }

  // Pretty output
  console.log(`\n╔══════════════════════════════════════════════╗`);
  console.log(`║  OKLCh Palette Generator — Brand Hue: ${brandHue}°    ║`);
  console.log(`╚══════════════════════════════════════════════╝\n`);

  for (const palette of [dark, light]) {
    const mode = palette.colorScheme.toUpperCase();
    console.log(`\n━━━ ${mode} THEME: ${palette.name} ━━━\n`);

    console.log("  Base surfaces:");
    console.log(`    base-100 (bg):      ${fmt(palette.base100)}`);
    console.log(`    base-200 (surface): ${fmt(palette.base200)}`);
    console.log(`    base-300 (border):  ${fmt(palette.base300)}`);
    console.log(`    base-content:       ${fmt(palette.baseContent)}`);

    console.log("\n  Brand colors:");
    console.log(`    primary:    ${fmt(palette.primary)}`);
    console.log(`    secondary:  ${fmt(palette.secondary)}  (H ${palette.secondary.h.toFixed(0)}° = brand−40°)`);
    console.log(`    accent:     ${fmt(palette.accent)}  (H ${palette.accent.h.toFixed(0)}° = brand+150°)`);
    console.log(`    neutral:    ${fmt(palette.neutral)}`);

    console.log("\n  Semantic colors:");
    console.log(`    info:    ${fmt(palette.info)}  (H=235°)`);
    console.log(`    success: ${fmt(palette.success)}  (H=150°)`);
    console.log(`    warning: ${fmt(palette.warning)}  (H=80°)`);
    console.log(`    error:   ${fmt(palette.error)}  (H=25°)`);

    console.log("\n  Content hierarchy:");
    console.log(`    strong:  ${fmt(palette.content.strong)}`);
    console.log(`    default: ${fmt(palette.content.default)}`);
    console.log(`    muted:   ${fmt(palette.content.muted)}`);
    console.log(`    subtle:  ${fmt(palette.content.subtle)}`);
    console.log(`    dim:     ${fmt(palette.content.dim)}`);

    console.log("\n  Surface tokens:");
    const s = palette.surfaces;
    console.log(`    card:     ${fmt(s.card.color, s.card.alpha)}`);
    console.log(`    elevated: ${fmt(s.elevated.color, s.elevated.alpha)}`);
    console.log(`    border:   ${fmt(s.border.color, s.border.alpha)}`);
    console.log(`    hover:    ${fmt(s.hover.color, s.hover.alpha)}`);

    printContrastReport(palette);
  }
}

main();
