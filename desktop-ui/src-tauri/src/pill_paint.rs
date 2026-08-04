//! Cairo renderer for the native Linux pill overlay.
//!
//! The Linux pill is drawn directly with cairo instead of a webview: WebKitGTK's
//! accelerated compositor drops the alpha channel on this stack (Nvidia
//! proprietary + GNOME/Mutter), so anything not fully opaque ghosts. Cairo never
//! touches EGL/dmabuf, so genuine translucency works — see
//! `docs/plans/pill-rendering.md`.
//!
//! Everything here is pure: a [`Frame`] in, drawing commands out. The window,
//! its state and its input live in `pill_native`.

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::LazyLock;

use gtk::cairo::{Context, LinearGradient, Operator};

// ── Colour ──────────────────────────────────────────────────────────────
//
// The theme is authored in oklch CSS custom properties. The pill webview never
// applied a theme (`pill-entry.tsx` sets no `data-theme`/`data-color-scheme`),
// so it always resolved `:root` — the dark token block plus the default `lune`
// DaisyUI theme. Those are the values mirrored below, and
// `tokens_match_the_stylesheets` reads the stylesheets back and fails if either
// side moves, which is what keeps the two platforms looking like one product.

/// A theme colour in oklch, as written in the stylesheets.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Oklch {
    l: f64,
    c: f64,
    /// Hue in degrees; ignored when chroma is 0.
    h: f64,
    alpha: f64,
}

/// Straight (non-premultiplied) sRGB with alpha, ready for `set_source_rgba`.
type Rgba = [f64; 4];

impl Oklch {
    const fn new(l: f64, c: f64, h: f64, alpha: f64) -> Self {
        Self { l, c, h, alpha }
    }

    const fn with_alpha(self, alpha: f64) -> Self {
        Self { alpha, ..self }
    }

    /// Oklab components — the space `color-mix(in oklab, …)` interpolates in.
    fn lab(self) -> (f64, f64, f64) {
        let (sin, cos) = self.h.to_radians().sin_cos();
        (self.l, self.c * cos, self.c * sin)
    }

    fn rgba(self) -> Rgba {
        let (l, a, b) = self.lab();
        lab_to_rgba(l, a, b, self.alpha)
    }
}

/// Oklab → sRGB (Björn Ottosson's matrices), gamma-encoded and clamped to gamut.
fn lab_to_rgba(l: f64, a: f64, b: f64, alpha: f64) -> Rgba {
    let l_ = (l + 0.396_337_777_4 * a + 0.215_803_757_3 * b).powi(3);
    let m_ = (l - 0.105_561_345_8 * a - 0.063_854_172_8 * b).powi(3);
    let s_ = (l - 0.089_484_177_5 * a - 1.291_485_548_0 * b).powi(3);
    let linear = [
        4.076_741_662_1 * l_ - 3.307_711_591_3 * m_ + 0.230_969_929_2 * s_,
        -1.268_438_004_6 * l_ + 2.609_757_401_1 * m_ - 0.341_319_396_5 * s_,
        -0.004_196_086_3 * l_ - 0.703_418_614_7 * m_ + 1.707_614_701_0 * s_,
    ];
    let encode = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        if x <= 0.003_130_8 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    };
    [
        encode(linear[0]),
        encode(linear[1]),
        encode(linear[2]),
        alpha.clamp(0.0, 1.0),
    ]
}

/// `color-mix(in oklab, base (1-t), tint t)` for two opaque colours.
fn mix(base: Oklch, tint: Oklch, t: f64) -> Rgba {
    let (bl, ba, bb) = base.lab();
    let (tl, ta, tb) = tint.lab();
    let lerp = |a: f64, b: f64| a + (b - a) * t;
    lab_to_rgba(lerp(bl, tl), lerp(ba, ta), lerp(bb, tb), 1.0)
}

const SURFACE_ELEVATED: Oklch = Oklch::new(0.22, 0.0, 0.0, 1.0);
const SURFACE_BORDER_STRONG: Oklch = Oklch::new(1.0, 0.0, 0.0, 0.16);
const BASE_CONTENT: Oklch = Oklch::new(0.94, 0.0, 0.0, 1.0);
const PRIMARY: Oklch = Oklch::new(0.74, 0.14, 295.0, 1.0);
const PRIMARY_CONTENT: Oklch = Oklch::new(0.16, 0.04, 295.0, 1.0);
const ACCENT: Oklch = Oklch::new(0.74, 0.10, 185.0, 1.0);
const ERROR: Oklch = Oklch::new(0.66, 0.18, 22.0, 1.0);
const ERROR_CONTENT: Oklch = Oklch::new(0.98, 0.005, 22.0, 1.0);
const WHITE: Oklch = Oklch::new(1.0, 0.0, 0.0, 1.0);
const BLACK: Oklch = Oklch::new(0.0, 0.0, 0.0, 1.0);

/// Resolved sRGB for every colour the pill paints. Built once — the theme can't
/// change under a running pill (the overlay has no theme switch of its own).
struct Palette {
    bar: Rgba,
    bar_hover: Rgba,
    bar_pressed: Rgba,
    border: Rgba,
    bevel_top: Rgba,
    bevel_bottom: Rgba,
    ring_listening: Rgba,
    ring_processing: Rgba,
    idle_mark: Rgba,
    dot: Rgba,
    pulse: Rgba,
    /// Sunk below the bar, so the destructive control reads as a hole in it.
    discard_chip: Rgba,
    discard_chip_hover: Rgba,
    discard_icon: Rgba,
    discard_icon_hover: Rgba,
    /// Raised to the full accent, so the confirming control reads as the one
    /// thing to press. One chip below the bar and one above it is the contrast
    /// the two controls are told apart by, before either icon is read.
    accept_chip: Rgba,
    accept_chip_hover: Rgba,
    accept_icon: Rgba,
}

static PALETTE: LazyLock<Palette> = LazyLock::new(|| {
    // Lifted off the base surface so the bar reads as an object over arbitrary
    // desktop content rather than a hole in it. The interactive fills mix into
    // that lightened colour, as the CSS mixes into `--pill-fill`.
    let bar = mix(SURFACE_ELEVATED, WHITE, 0.16);
    let bar_oklch = mix_back(bar);
    Palette {
        bar,
        bar_hover: mix(bar_oklch, PRIMARY, 0.12),
        bar_pressed: mix(bar_oklch, PRIMARY, 0.22),
        border: SURFACE_BORDER_STRONG.rgba(),
        bevel_top: WHITE.with_alpha(0.16).rgba(),
        bevel_bottom: BLACK.with_alpha(0.20).rgba(),
        ring_listening: PRIMARY.with_alpha(0.62).rgba(),
        ring_processing: ACCENT.with_alpha(0.62).rgba(),
        idle_mark: PRIMARY.with_alpha(0.70).rgba(),
        dot: PRIMARY.rgba(),
        pulse: ACCENT.rgba(),
        discard_chip: mix(bar_oklch, BLACK, 0.5),
        discard_chip_hover: ERROR.rgba(),
        discard_icon: BASE_CONTENT.with_alpha(0.75).rgba(),
        discard_icon_hover: ERROR_CONTENT.rgba(),
        accept_chip: PRIMARY.rgba(),
        accept_chip_hover: mix(PRIMARY, WHITE, 0.18),
        accept_icon: PRIMARY_CONTENT.rgba(),
    }
});

/// sRGB → oklch, so a derived fill can be mixed again the way the CSS chains
/// `color-mix(… var(--pill-fill) …)`.
fn mix_back(rgba: Rgba) -> Oklch {
    let decode = |x: f64| {
        if x <= 0.040_45 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (decode(rgba[0]), decode(rgba[1]), decode(rgba[2]));
    let l = (0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b).cbrt();
    let m = (0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b).cbrt();
    let s = (0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b).cbrt();
    let lab = (
        0.210_454_255_3 * l + 0.793_617_785_0 * m - 0.004_072_046_8 * s,
        1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s,
        0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s,
    );
    Oklch::new(
        lab.0,
        lab.1.hypot(lab.2),
        lab.2.atan2(lab.1).to_degrees().rem_euclid(360.0),
        rgba[3],
    )
}

// ── Geometry ────────────────────────────────────────────────────────────

/// Bar border width (logical px).
const BORDER: f64 = 2.0;
/// Diameter of the drawn end-control chips.
const CHIP: f64 = 28.0;
/// Rendered size of the 24×24 icon viewBoxes.
const CAP_ICON: f64 = 15.0;

const IDLE_MARK_W: f64 = 26.0;
const IDLE_MARK_H: f64 = 6.0;

/// Level-meter bars: centre bars react most, edges least. An odd count so the
/// arch has a single peak — an even one puts two equal bars at the centre and
/// flattens the top of every shape the meter makes.
const DOT_WEIGHTS: [f64; 7] = [0.35, 0.6, 0.85, 1.0, 0.85, 0.6, 0.35];
/// Bar width taken from the brand mark (`MARK_WAVE` is 6 wide), so the meter
/// and the logo read as one visual language. The mark's 3-unit gap is too tight
/// at this height — the bars merge into a blob — so the gap is opened to 4.
const DOT_W: f64 = 6.0;
const DOT_GAP: f64 = 4.0;
/// Resting height of the centre bars. The edges taper by weight, so the band
/// keeps a waveform silhouette even in near-silence — where the smoothed level
/// sits just above zero and a uniform height would render a flat picket fence.
const DOT_H: f64 = 11.0;
/// Opacity floor, shared with the processing pulse: below it the mark all but
/// vanishes on the lightened bar.
const DOT_MIN_ALPHA: f64 = 0.55;
/// Idle-bounce offset between one symmetric pair of bars and the next, so the
/// resting animation ripples outward from the middle instead of reading as a
/// left-to-right marquee (which fights the centre-weighted shape a live level
/// produces).
const DOT_STAGGER: f64 = 0.110;
const DOT_BOUNCE_PERIOD: f64 = 1.2;

const PULSE_DIAMETER: f64 = 14.0;
const PULSE_PERIOD: f64 = 1.3;

/// The overlay sits back a touch from whatever is behind it. Mirrored by
/// `--pill-opacity` in components.css.
const WINDOW_ALPHA: f64 = 0.975;

/// The two end controls. Both exist only while listening: idle has nothing to
/// discard or accept, and processing can't be interrupted mid-finalise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cap {
    /// Right, or bottom when vertical. Throws the utterance away.
    Discard,
    /// Left, or top when vertical. Stops and transcribes.
    Accept,
}

/// What the pill is showing. Mirrors the service status the runtime publishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Idle,
    Listening,
    Processing,
}

impl Mode {
    pub fn from_status(status: &str) -> Self {
        match status {
            "listening" => Self::Listening,
            "processing" => Self::Processing,
            _ => Self::Idle,
        }
    }
}

/// Everything one repaint needs. Built by `pill_native` from the live config
/// and UI state.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub w: f64,
    pub h: f64,
    pub radius: f64,
    /// Caps move to the top/bottom and the status content rotates 90°.
    pub vertical: bool,
    pub mode: Mode,
    /// Smoothed mic level, 0.0–1.0. Drives the bars while listening.
    pub level: f64,
    /// Seconds since the pill was shown — the clock every loop reads.
    pub elapsed: f64,
    /// Content entrance progress, 0.0–1.0, restarted when the mode changes.
    pub swap: f64,
    pub hover: bool,
    pub pressed: bool,
    /// Which end control the pointer is over, if any.
    pub hovered_cap: Option<Cap>,
    /// Whole-pill opacity, driven down by the discard flash.
    pub fade: f64,
}

pub fn draw(cr: &Context, f: &Frame) {
    // The window is an ARGB surface we own outright: clear it rather than
    // painting a background, so everything outside the bar is truly empty.
    cr.set_operator(Operator::Clear);
    let _ = cr.paint();
    cr.set_operator(Operator::Over);

    // Composite through a group so the whole overlay fades as one image. Done
    // per colour instead, the bar would show through its own border and the
    // logo would show through the bar.
    cr.push_group();

    draw_bar(cr, f);
    draw_content(cr, f);
    draw_cap(cr, f, Cap::Discard);
    draw_cap(cr, f, Cap::Accept);

    let _ = cr.pop_group_to_source();
    let _ = cr.paint_with_alpha(f.fade * WINDOW_ALPHA);
}

fn draw_bar(cr: &Context, f: &Frame) {
    // The border straddles the path, so inset by half of it to keep the outer
    // edge exactly on the window footprint.
    let i = BORDER / 2.0;
    rounded_rect(cr, i, i, f.w - BORDER, f.h - BORDER, f.radius - i);

    let fill = if f.pressed {
        PALETTE.bar_pressed
    } else if f.hover && f.mode != Mode::Processing {
        PALETTE.bar_hover
    } else {
        PALETTE.bar
    };
    set_source(cr, fill);
    let _ = cr.fill_preserve();

    cr.set_line_width(BORDER);
    match f.mode {
        // A live state rings the whole bar in the colour its contents use, so
        // "it's on" is readable from across the room and not only from the mark
        // in the middle. One edge treatment at a time: the ring replaces the
        // bevel rather than layering over it.
        Mode::Listening => set_source(cr, PALETTE.ring_listening),
        Mode::Processing => set_source(cr, PALETTE.ring_processing),
        Mode::Idle => set_source(cr, PALETTE.border),
    }
    let _ = cr.stroke_preserve();

    if f.mode == Mode::Idle {
        // Bevel: light along the top edge, dark along the bottom, so the bar
        // reads as a raised object. A vertical gradient rather than the
        // webview's per-edge border colours — CSS cannot gradient a border, and
        // the hard switch at the corners was the compromise that came from that.
        let bevel = LinearGradient::new(0.0, 0.0, 0.0, f.h);
        add_stop(&bevel, 0.0, PALETTE.bevel_top);
        add_stop(&bevel, 0.5, [0.0, 0.0, 0.0, 0.0]);
        add_stop(&bevel, 1.0, PALETTE.bevel_bottom);
        let _ = cr.set_source(&bevel);
    }
    let _ = cr.stroke();
}

/// Distance from the bar's end to a chip's centre: halfway between the bar's
/// inner edge and the level meter, so each control sits in the middle of the
/// space it has instead of crowding the rounded end. Clamped so the chip stays
/// inside the border on a pill too short to give it that room.
fn cap_inset(f: &Frame) -> f64 {
    let extent = if f.vertical { f.h } else { f.w };
    let band_edge = (extent - band_span()) / 2.0;
    ((BORDER + band_edge) / 2.0).max(BORDER + CHIP / 2.0)
}

/// How far in from the bar's end a cap's hit box reaches — to the far side of
/// its chip. Everything outward of that belongs to the control, so the rounded
/// end is part of the target rather than a dead margin.
fn cap_reach(f: &Frame) -> f64 {
    cap_inset(f) + CHIP / 2.0
}

/// Centre of a cap's chip, in window coordinates.
fn chip_centre(f: &Frame, cap: Cap) -> (f64, f64) {
    let inset = cap_inset(f);
    let extent = if f.vertical { f.h } else { f.w };
    let at = match cap {
        Cap::Accept => inset,
        Cap::Discard => extent - inset,
    };
    if f.vertical {
        (f.w / 2.0, at)
    } else {
        (at, f.h / 2.0)
    }
}

fn draw_cap(cr: &Context, f: &Frame, cap: Cap) {
    if !caps_live(f) {
        return;
    }
    let hovered = f.hovered_cap == Some(cap);
    let (chip, icon) = match (cap, hovered) {
        (Cap::Discard, false) => (PALETTE.discard_chip, PALETTE.discard_icon),
        (Cap::Discard, true) => (PALETTE.discard_chip_hover, PALETTE.discard_icon_hover),
        (Cap::Accept, false) => (PALETTE.accept_chip, PALETTE.accept_icon),
        (Cap::Accept, true) => (PALETTE.accept_chip_hover, PALETTE.accept_icon),
    };

    let _ = cr.save();
    let (cx, cy) = chip_centre(f, cap);
    cr.translate(cx, cy);
    // Entrance: the chips ride the same content swap as the waveform, so the
    // three appear as one change of state rather than three.
    let s = 0.92 + 0.08 * f.swap;
    cr.scale(s, s);

    set_source(cr, alpha_scaled(chip, f.swap));
    cr.arc(0.0, 0.0, CHIP / 2.0, 0.0, TAU);
    let _ = cr.fill();

    set_source(cr, alpha_scaled(icon, f.swap));
    let k = CAP_ICON / 24.0;
    cr.scale(k, k);
    cr.translate(-12.0, -12.0);
    cr.set_line_width(2.4);
    cr.set_line_cap(gtk::cairo::LineCap::Round);
    cr.set_line_join(gtk::cairo::LineJoin::Round);
    match cap {
        Cap::Discard => {
            cr.move_to(18.0, 6.0);
            cr.line_to(6.0, 18.0);
            cr.move_to(6.0, 6.0);
            cr.line_to(18.0, 18.0);
        }
        Cap::Accept => {
            cr.move_to(12.0, 19.0);
            cr.line_to(12.0, 5.0);
            cr.move_to(5.0, 12.0);
            cr.line_to(12.0, 5.0);
            cr.line_to(19.0, 12.0);
        }
    }
    let _ = cr.stroke();

    let _ = cr.restore();
}

fn draw_content(cr: &Context, f: &Frame) {
    let _ = cr.save();
    // The status content is centred in the whole bar, not in the space between
    // the caps — the caps are overlays, matching the webview's full-bleed
    // surface.
    cr.translate(f.w / 2.0, f.h / 2.0);
    if f.vertical {
        cr.rotate(FRAC_PI_2);
    }
    // Entrance: scale up from 0.92 as the new content fades in, so a mode
    // change reads as a swap rather than a jump.
    let s = 0.92 + 0.08 * f.swap;
    cr.scale(s, s);

    match f.mode {
        Mode::Idle => {
            // A static "ready" mark. The webview breathed it on a 3.2s loop;
            // that bought nothing the pill's mere presence doesn't already say,
            // and it cost a permanent 60 fps repaint loop for an overlay that
            // is idle almost all the time.
            set_source(cr, alpha_scaled(PALETTE.idle_mark, f.swap));
            rounded_rect(
                cr,
                -IDLE_MARK_W / 2.0,
                -IDLE_MARK_H / 2.0,
                IDLE_MARK_W,
                IDLE_MARK_H,
                IDLE_MARK_H / 2.0,
            );
            let _ = cr.fill();
        }
        Mode::Listening => draw_dots(cr, f),
        Mode::Processing => {
            let beat = (1.0 - (TAU * f.elapsed / PULSE_PERIOD).cos()) / 2.0;
            // Opacity floors at 0.55: below that the mark all but vanishes on
            // the lightened bar and "processing" looks like a dead pill.
            let r = PULSE_DIAMETER / 2.0 * (0.8 + 0.3 * beat);
            set_source(
                cr,
                alpha_scaled(PALETTE.pulse, f.swap * (0.55 + 0.45 * beat)),
            );
            cr.arc(0.0, 0.0, r, 0.0, TAU);
            let _ = cr.fill();
        }
    }

    let _ = cr.restore();
}

/// Width of the whole level meter. The caps are placed against it, so it has to
/// be knowable without drawing it.
fn band_span() -> f64 {
    DOT_WEIGHTS.len() as f64 * DOT_W + (DOT_WEIGHTS.len() - 1) as f64 * DOT_GAP
}

fn draw_dots(cr: &Context, f: &Frame) {
    let span = band_span();
    for (i, weight) in DOT_WEIGHTS.iter().enumerate() {
        let (scale, opacity) = if f.level > 0.0 {
            (1.0 + f.level * weight * 2.0, dot_alpha(f.level))
        } else {
            // Silence: the resting bounce, staggered outward from the centre so
            // idle and speaking read as the same widget.
            let centre = (DOT_WEIGHTS.len() - 1) as f64 / 2.0;
            let delay = ((i as f64 - centre).abs()).floor() * DOT_STAGGER;
            bounce((f.elapsed - delay) / DOT_BOUNCE_PERIOD)
        };
        let x = -span / 2.0 + i as f64 * (DOT_W + DOT_GAP);
        let h = bar_height(*weight) * scale;
        set_source(cr, alpha_scaled(PALETTE.dot, f.swap * opacity));
        rounded_rect(cr, x, -h / 2.0, DOT_W, h, DOT_W / 2.0);
        let _ = cr.fill();
    }
}

/// Resting height of one bar: full height at the centre, tapering to half the
/// taper at the edges. The same weight drives height and response, so a bar
/// that reacts less also sits lower and the band never looks flat.
fn bar_height(weight: f64) -> f64 {
    DOT_H * (1.0 + weight) / 2.0
}

/// One cycle of the resting bounce: up over the first 40%, back down by 80%,
/// flat until the loop restarts. Returns (scaleY, opacity).
fn bounce(phase: f64) -> (f64, f64) {
    let t = phase.rem_euclid(1.0);
    let k = if t < 0.4 {
        ease_in_out(t / 0.4)
    } else if t < 0.8 {
        ease_in_out(1.0 - (t - 0.4) / 0.4)
    } else {
        0.0
    };
    (1.0 + 1.1 * k, dot_alpha(k))
}

/// The band's one opacity ramp, driven by the live level or by the resting
/// bounce. Sharing it keeps the two from stepping against each other at the
/// moment the first level arrives.
fn dot_alpha(x: f64) -> f64 {
    DOT_MIN_ALPHA + (1.0 - DOT_MIN_ALPHA) * x
}

fn ease_in_out(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Primitives ──────────────────────────────────────────────────────────

fn rounded_rect(cr: &Context, x: f64, y: f64, w: f64, h: f64, r: f64) {
    let r = r.clamp(0.0, (w / 2.0).min(h / 2.0));
    cr.new_sub_path();
    cr.arc(x + w - r, y + r, r, -FRAC_PI_2, 0.0);
    cr.arc(x + w - r, y + h - r, r, 0.0, FRAC_PI_2);
    cr.arc(x + r, y + h - r, r, FRAC_PI_2, PI);
    cr.arc(x + r, y + r, r, PI, PI + FRAC_PI_2);
    cr.close_path();
}

fn set_source(cr: &Context, c: Rgba) {
    cr.set_source_rgba(c[0], c[1], c[2], c[3]);
}

fn add_stop(g: &LinearGradient, offset: f64, c: Rgba) {
    g.add_color_stop_rgba(offset, c[0], c[1], c[2], c[3]);
}

fn alpha_scaled(c: Rgba, k: f64) -> Rgba {
    [c[0], c[1], c[2], c[3] * k.clamp(0.0, 1.0)]
}

/// Whether the end controls are on screen at all. One rule, read by both the
/// renderer and the hit test, so an invisible chip can never be clickable.
fn caps_live(f: &Frame) -> bool {
    f.mode == Mode::Listening
}

/// Which end control is at (x, y) in window coordinates, if any. Shared with
/// the renderer so the hit boxes and the drawn chips can't drift apart.
pub fn cap_at(f: &Frame, x: f64, y: f64) -> Option<Cap> {
    if !caps_live(f) {
        return None;
    }
    let (along, extent) = if f.vertical { (y, f.h) } else { (x, f.w) };
    let reach = cap_reach(f);
    if along <= reach {
        Some(Cap::Accept)
    } else if along >= extent - reach {
        Some(Cap::Discard)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oklch literals above are copies of CSS custom properties. Read the
    /// stylesheets back and fail if either side moved: without this the two
    /// pill implementations drift apart the first time the theme changes, and
    /// nothing would report it.
    #[test]
    fn tokens_match_the_stylesheets() {
        let styles = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/styles/");
        // The pill window applies no theme, so it resolves the first (dark /
        // `lune`) declaration in each file — which is what `first_decl` reads.
        let tokens = std::fs::read_to_string(format!("{styles}tokens.css")).unwrap();
        let daisy = std::fs::read_to_string(format!("{styles}daisyui.css")).unwrap();
        let cases = [
            (&tokens, "--surface-elevated", SURFACE_ELEVATED),
            (&tokens, "--surface-border-strong", SURFACE_BORDER_STRONG),
            (&daisy, "--color-base-content", BASE_CONTENT),
            (&daisy, "--color-primary", PRIMARY),
            (&daisy, "--color-primary-content", PRIMARY_CONTENT),
            (&daisy, "--color-accent", ACCENT),
            (&daisy, "--color-error", ERROR),
            (&daisy, "--color-error-content", ERROR_CONTENT),
        ];
        for (css, name, expected) in cases {
            assert_eq!(first_decl(css, name), Some(expected), "{name}");
        }
    }

    /// The first `--name: oklch(L C H[ / A]);` declaration in a stylesheet.
    fn first_decl(css: &str, name: &str) -> Option<Oklch> {
        let line = css
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with(&format!("{name}:")))?;
        let inner = line.split_once("oklch(")?.1.split_once(')')?.0;
        let (coords, alpha) = match inner.split_once('/') {
            Some((c, a)) => (c, a.trim().parse().ok()?),
            None => (inner, 1.0),
        };
        let mut parts = coords.split_whitespace().map(|p| p.parse::<f64>().ok());
        Some(Oklch::new(
            parts.next()??,
            parts.next()??,
            parts.next()??,
            alpha,
        ))
    }

    /// The level meter is drawn twice — here and in `dot-style.ts` — from the
    /// same numbers. Same argument as the colour tokens: read the other side
    /// back rather than trusting two hand-kept copies to stay equal.
    #[test]
    fn the_band_matches_the_webview_meter() {
        let ts = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../src/features/dictation/dot-style.ts"
        ))
        .unwrap();
        let decl = |name: &str| -> String {
            ts.lines()
                .map(str::trim)
                .find_map(|l| {
                    l.trim_start_matches("export ")
                        .strip_prefix(&format!("const {name} = "))?
                        .strip_suffix(';')
                })
                .unwrap_or_else(|| panic!("{name} not found"))
                .to_owned()
        };
        assert_eq!(decl("DOT_COUNT"), DOT_WEIGHTS.len().to_string());
        assert_eq!(decl("DOT_WEIGHTS"), format!("{DOT_WEIGHTS:?}"));
        let stagger_ms: f64 = decl("DOT_STAGGER_MS").parse().unwrap();
        assert!(
            (stagger_ms / 1000.0 - DOT_STAGGER).abs() < 1e-9,
            "{stagger_ms}"
        );
        assert_eq!(decl("DOT_HEIGHT_PX"), DOT_H.to_string());
        assert_eq!(decl("DOT_MIN_ALPHA"), DOT_MIN_ALPHA.to_string());
    }

    #[test]
    fn neutral_tokens_round_trip_to_grey() {
        let [r, g, b, a] = SURFACE_ELEVATED.rgba();
        assert!((r - g).abs() < 1e-6 && (g - b).abs() < 1e-6, "{r} {g} {b}");
        assert_eq!(a, 1.0);
        // oklch L=1 is white, L=0 black — the two ends the bevel mixes toward.
        assert!(WHITE.rgba()[0] > 0.999);
        assert!(BLACK.rgba()[0] < 0.001);
    }

    #[test]
    fn oklch_survives_a_round_trip_through_srgb() {
        for c in [PRIMARY, ACCENT, ERROR, SURFACE_ELEVATED] {
            let back = mix_back(c.rgba());
            assert!((back.l - c.l).abs() < 1e-3, "L {back:?} vs {c:?}");
            assert!((back.c - c.c).abs() < 1e-3, "C {back:?} vs {c:?}");
            if c.c > 0.0 {
                assert!((back.h - c.h).abs() < 0.5, "H {back:?} vs {c:?}");
            }
        }
    }

    #[test]
    fn mixing_toward_white_lightens_without_tinting() {
        let bar = PALETTE.bar;
        let base = SURFACE_ELEVATED.rgba();
        assert!(bar[0] > base[0], "{bar:?} vs {base:?}");
        assert!((bar[0] - bar[1]).abs() < 1e-6 && (bar[1] - bar[2]).abs() < 1e-6);
        // The pressed fill carries more of the accent than the hover fill.
        assert!(PALETTE.bar_pressed[2] > PALETTE.bar_hover[2]);
        assert!(PALETTE.bar_hover[2] > PALETTE.bar[2]);
    }

    #[test]
    fn the_resting_bounce_is_a_loop_that_rests() {
        let (scale_lo, op_lo) = bounce(0.0);
        assert!((scale_lo - 1.0).abs() < 1e-9 && (op_lo - DOT_MIN_ALPHA).abs() < 1e-9);
        let (scale_hi, op_hi) = bounce(0.4);
        assert!((scale_hi - 2.1).abs() < 1e-9 && (op_hi - 1.0).abs() < 1e-9);
        // Flat tail, and the same point one period later.
        assert_eq!(bounce(0.9), (1.0, DOT_MIN_ALPHA));
        assert_eq!(bounce(1.4), bounce(0.4));
        assert_eq!(bounce(-0.6), bounce(0.4));
    }

    #[test]
    fn the_hit_boxes_follow_the_orientation() {
        let f = listening_frame();
        assert_eq!(cap_at(&f, 10.0, 25.0), Some(Cap::Accept));
        assert_eq!(cap_at(&f, 160.0, 25.0), Some(Cap::Discard));
        assert_eq!(cap_at(&f, 84.0, 25.0), None);
        let v = Frame {
            w: 50.0,
            h: 168.0,
            vertical: true,
            ..f
        };
        assert_eq!(cap_at(&v, 25.0, 10.0), Some(Cap::Accept));
        assert_eq!(cap_at(&v, 25.0, 160.0), Some(Cap::Discard));
        assert_eq!(cap_at(&v, 25.0, 84.0), None);
    }

    /// Nothing to discard or accept outside a live session, and a chip that
    /// isn't drawn must not be clickable either.
    #[test]
    fn the_caps_exist_only_while_listening() {
        for mode in [Mode::Idle, Mode::Processing] {
            let f = Frame {
                mode,
                ..listening_frame()
            };
            assert_eq!(cap_at(&f, 10.0, 25.0), None, "{mode:?}");
            assert_eq!(cap_at(&f, 160.0, 25.0), None, "{mode:?}");
        }
    }

    /// Each chip sits inside the bar and clear of the level meter, centred in
    /// the space between the two.
    #[test]
    fn the_chips_sit_between_the_end_and_the_meter() {
        let f = listening_frame();
        let band_edge = (f.w - band_span()) / 2.0;
        for cap in [Cap::Discard, Cap::Accept] {
            let (cx, _) = chip_centre(&f, cap);
            let near = if cap == Cap::Accept { cx } else { f.w - cx };
            assert!(near - CHIP / 2.0 >= BORDER, "{cap:?} overhangs the end");
            assert!(near + CHIP / 2.0 <= band_edge, "{cap:?} crowds the meter");
            // Equidistant from the bar's inner edge and the meter.
            assert!(
                ((near - BORDER) - (band_edge - near)).abs() < 1e-9,
                "{cap:?} is not centred in its space"
            );
        }
    }

    /// A pill too short to centre the chips must still keep them inside the
    /// border, even at the cost of crowding the meter.
    #[test]
    fn a_cramped_pill_keeps_its_chips_inside() {
        let f = Frame {
            w: 96.0,
            ..listening_frame()
        };
        assert!(cap_inset(&f) - CHIP / 2.0 >= BORDER);
    }

    fn listening_frame() -> Frame {
        Frame {
            w: 168.0,
            h: 50.0,
            radius: 15.0,
            vertical: false,
            mode: Mode::Listening,
            level: 0.0,
            elapsed: 0.0,
            swap: 1.0,
            hover: false,
            pressed: false,
            hovered_cap: None,
            fade: 1.0,
        }
    }

    #[test]
    fn status_maps_to_the_three_modes() {
        assert_eq!(Mode::from_status("listening"), Mode::Listening);
        assert_eq!(Mode::from_status("processing"), Mode::Processing);
        for idle in ["ready", "error", "initializing", ""] {
            assert_eq!(Mode::from_status(idle), Mode::Idle);
        }
    }
}
