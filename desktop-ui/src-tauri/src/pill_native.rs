// SPDX-License-Identifier: GPL-3.0-or-later

//! The Linux dictation pill: a bare GTK window painted with cairo.
//!
//! Not a Tauri window. Tauri only creates windows with a `WebviewWindowBuilder`,
//! so every one of them carries a `WebKitWebView` — and on this stack that
//! webview cannot composite alpha (see `docs/plans/pill-rendering.md`). The pill
//! needs no web content, no IPC and no window features Tauri adds, so it is
//! created directly against GTK, which tao (Tauri's windowing layer on Linux)
//! *is* rather than merely wraps: there is one GTK main loop, already running by
//! the time `setup` calls in here.
//!
//! Being in-process also removes the event bus from the loop — status, level and
//! settings arrive by direct call instead of `emit_to("pill", …)`, and clicks
//! reach the runtime without an IPC round trip.
//!
//! Drawing lives in `pill_paint`; this module owns the window, its state and its
//! input. Every function here must run on the main thread — GTK is not `Send`,
//! and the pill is held in a thread-local for exactly that reason. Callers on
//! other threads go through `crate::pill`, which hops via `run_on_main_thread`.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use gtk::gdk;
use gtk::glib;
use gtk::prelude::*;
use gtk_layer_shell::LayerShell;
use tauri::{AppHandle, Manager};

use crate::pill::{MonitorRect, PillOverlayConfig};
use crate::pill_paint::{self, Cap, Frame, Mode, cap_at};

/// GDK's primary (left) mouse button. The pill responds to nothing else.
const PRIMARY_BUTTON: u32 = 1;
/// Pointer movement (px) beyond which a press counts as a drag rather than a
/// click. Below it, release still toggles dictation.
const DRAG_THRESHOLD: f64 = 3.0;
/// Cross-fade when the pill's contents swap between states.
const CONTENT_SWAP: Duration = Duration::from_millis(180);
/// Acknowledgement fade after the discard control is activated.
const CANCEL_FLASH: Duration = Duration::from_millis(280);

/// How long "processing" must persist before the pill shows it, and how long it
/// stays once shown. An utterance with no speech in it is trimmed to nothing by
/// the VAD and decodes instantly, so `capture.rs` can publish and withdraw the
/// state inside a single frame — a flash of a different hue carrying no
/// information, which reads as a glitch. A state too brief to be read is not
/// drawn at all; one worth drawing stays long enough to see. Only processing is
/// gated: listening is the feedback for the hotkey and must be immediate.
const PROCESSING_DEBUT: Duration = Duration::from_millis(150);
const PROCESSING_DWELL: Duration = Duration::from_millis(250);

/// Fast-attack / slow-release smoothing for the mic level. Applied where the
/// level *arrives* (~15 Hz from the pipeline), not per frame, so the response
/// curve doesn't change with the display's refresh rate.
const LEVEL_ATTACK: f32 = 0.35;
const LEVEL_RELEASE: f32 = 0.15;

/// Live service status, written from the runtime's async tasks. Encoded as a
/// [`Mode`] discriminant so the value can cross threads without a lock.
static STATUS: AtomicU8 = AtomicU8::new(MODE_IDLE);
const MODE_IDLE: u8 = 0;
const MODE_LISTENING: u8 = 1;
const MODE_PROCESSING: u8 = 2;

/// Smoothed mic level as `f32` bits — the only state updated at audio rate, so
/// it is written without a main-thread hop; the running tick callback picks it
/// up on the next frame.
static LEVEL: AtomicU32 = AtomicU32::new(0);

thread_local! {
    static PILL: RefCell<Option<Pill>> = const { RefCell::new(None) };
}

struct Pill {
    window: gtk::Window,
    area: gtk::DrawingArea,
    ui: Rc<RefCell<Ui>>,
}

/// Everything the renderer reads that isn't the live config. Owned by the main
/// thread, shared between the draw handler, the tick callback and the input
/// handlers.
struct Ui {
    cfg: PillOverlayConfig,
    mode: Mode,
    /// Animation clock, restarted on every show so a session's animations begin
    /// at a known phase.
    started: Instant,
    /// When the current contents appeared — drives the cross-fade, and dates
    /// the dwell that keeps a shown state on screen long enough to read.
    swap_at: Instant,
    /// The status as last published by the runtime, and when it changed. Not
    /// the same as `mode`, which lags it while the dwell rule holds.
    published: Mode,
    published_at: Instant,
    hover: bool,
    hovered_cap: Option<Cap>,
    /// When the discard acknowledgement started, if one is running.
    flash_at: Option<Instant>,
    press: Option<Press>,
    ticking: bool,
}

/// An in-flight pointer press. Both anchors are recorded because the two
/// positioning paths need different ones: on X11 the pointer's root position is
/// absolute, so the drag can be re-derived from it every event and cannot
/// accumulate rounding drift; layer-shell surfaces have no global pointer, so
/// there the window-relative position is measured against the grab point in the
/// *current* window frame, which is self-correcting for the same reason.
struct Press {
    root: (f64, f64),
    local: (f64, f64),
    origin: (i32, i32),
    dragging: bool,
    /// Which end control the press landed on, if any.
    cap: Option<Cap>,
}

impl Ui {
    fn frame(&self) -> Frame {
        let now = Instant::now();
        let progress = |since: Instant, span: Duration| {
            (now.duration_since(since).as_secs_f64() / span.as_secs_f64()).clamp(0.0, 1.0)
        };
        let pressed = self.press.as_ref().is_some_and(|p| !p.dragging);
        Frame {
            w: f64::from(self.cfg.w),
            h: f64::from(self.cfg.h),
            radius: f64::from(self.cfg.radius),
            vertical: self.cfg.h > self.cfg.w,
            mode: self.mode,
            level: f64::from(f32::from_bits(LEVEL.load(Ordering::Acquire))),
            elapsed: now.duration_since(self.started).as_secs_f64(),
            swap: progress(self.swap_at, CONTENT_SWAP),
            // The caps sit over the surface, so hovering one is not hovering
            // the surface.
            hover: self.hover && self.hovered_cap.is_none(),
            pressed: pressed && self.hovered_cap.is_none(),
            hovered_cap: self.hovered_cap,
            fade: self
                .flash_at
                .map_or(1.0, |at| 1.0 - progress(at, CANCEL_FLASH)),
        }
    }

    /// Whether anything on screen is still moving. When nothing is, the tick
    /// callback stops and the pill costs nothing until something happens.
    fn animating(&self, f: &Frame) -> bool {
        self.mode != Mode::Idle
            || f.swap < 1.0
            || self.flash_at.is_some()
            // A published state the dwell rule is still withholding.
            || self.mode != self.published
    }

    /// Advance the time-based state one frame. Returns whether to keep ticking.
    fn advance(&mut self) -> bool {
        let now = Instant::now();
        let published = mode_from_status();
        if published != self.published {
            self.published = published;
            self.published_at = now;
        }

        let mode = if published == Mode::Processing
            && now.duration_since(self.published_at) < PROCESSING_DEBUT
        {
            // Too new to know whether it will last long enough to be worth
            // showing. If it is withdrawn first, it is never drawn.
            self.mode
        } else if self.mode == Mode::Processing
            && now.duration_since(self.swap_at) < PROCESSING_DWELL
        {
            self.mode
        } else {
            published
        };

        if mode != self.mode {
            self.mode = mode;
            self.swap_at = now;
            if mode != Mode::Listening {
                LEVEL.store(0f32.to_bits(), Ordering::Release);
            }
        }

        if self
            .flash_at
            .is_some_and(|at| now.duration_since(at) >= CANCEL_FLASH)
        {
            self.flash_at = None;
        }

        let f = self.frame();
        self.animating(&f)
    }
}

fn mode_from_status() -> Mode {
    match STATUS.load(Ordering::Acquire) {
        MODE_LISTENING => Mode::Listening,
        MODE_PROCESSING => Mode::Processing,
        _ => Mode::Idle,
    }
}

// ── Lifecycle ───────────────────────────────────────────────────────────

pub fn exists() -> bool {
    PILL.with_borrow(Option::is_some)
}

/// Create the pill window, hidden. Idempotent. Main thread only.
pub fn create(app: &AppHandle) {
    if exists() {
        return;
    }
    let cfg = crate::pill::pill_cfg();
    let window = gtk::Window::new(gtk::WindowType::Toplevel);
    // The window name is what AT-SPI reports for the overlay. A cairo surface
    // has no accessibility tree of its own, and the pill has no focusable
    // controls, so this is the whole of its accessible identity.
    window.set_title("Lune Voice dictation");
    window.set_decorated(false);
    window.set_resizable(false);
    window.set_accept_focus(false);
    window.set_focus_on_map(false);
    window.set_skip_taskbar_hint(true);
    window.set_skip_pager_hint(true);
    window.set_type_hint(gdk::WindowTypeHint::Utility);
    window.set_keep_above(true);

    // The 32-bit ARGB visual is what makes real transparency possible: cairo
    // then draws through XRender into a server-side ARGB drawable, never
    // touching the EGL/dmabuf path that loses the alpha channel here.
    match gdk::Screen::default().and_then(|s| s.rgba_visual()) {
        Some(visual) => window.set_visual(Some(&visual)),
        None => tracing::warn!("no RGBA visual — the pill will paint on an opaque background"),
    }
    window.set_app_paintable(true);
    window.set_size_request(cfg.w, cfg.h);
    window.set_default_size(cfg.w, cfg.h);

    if use_layer_shell() {
        // Layer-shell surfaces can't be moved by the compositor, so absolute
        // placement (and dragging) is one anchored corner plus left/top margins.
        // Must be initialised before the window is realized.
        window.init_layer_shell();
        window.set_layer(gtk_layer_shell::Layer::Overlay);
        window.set_namespace("lune-voice-pill");
        window.set_keyboard_mode(gtk_layer_shell::KeyboardMode::None);
        window.set_anchor(gtk_layer_shell::Edge::Top, true);
        window.set_anchor(gtk_layer_shell::Edge::Left, true);
        // -1 = don't reserve exclusive space (don't push other windows around).
        window.set_exclusive_zone(-1);
    }

    let area = gtk::DrawingArea::new();
    // Without this GtkWidget paints the theme's background under our draw
    // handler, and the pill loses its transparency to a grey rectangle.
    area.set_app_paintable(true);
    area.add_events(
        gdk::EventMask::BUTTON_PRESS_MASK
            | gdk::EventMask::BUTTON_RELEASE_MASK
            | gdk::EventMask::POINTER_MOTION_MASK
            | gdk::EventMask::ENTER_NOTIFY_MASK
            | gdk::EventMask::LEAVE_NOTIFY_MASK,
    );
    window.add(&area);

    let now = Instant::now();
    let ui = Rc::new(RefCell::new(Ui {
        cfg,
        mode: mode_from_status(),
        started: now,
        swap_at: now,
        published: mode_from_status(),
        published_at: now,
        hover: false,
        hovered_cap: None,
        flash_at: None,
        press: None,
        ticking: false,
    }));

    area.connect_draw({
        let ui = ui.clone();
        move |_, cr| {
            pill_paint::draw(cr, &ui.borrow().frame());
            glib::Propagation::Proceed
        }
    });
    connect_input(&area, &ui, app);

    PILL.set(Some(Pill { window, area, ui }));
}

/// Destroy the pill window. Main thread only.
pub fn destroy() {
    if let Some(pill) = PILL.take() {
        pill.window.close();
    }
}

/// Map the pill and restart its animation clock. Main thread only.
pub fn show() {
    with_pill(|pill| {
        place(pill);
        {
            let mut ui = pill.ui.borrow_mut();
            let now = Instant::now();
            ui.started = now;
            ui.mode = mode_from_status();
            ui.published = ui.mode;
            ui.published_at = now;
            ui.swap_at = now;
        }
        pill.window.show_all();
        // Follow the user across workspaces instead of living on the one it was
        // mapped on — the pill is a global dictation overlay. Under X11 this is
        // `_NET_WM_STATE_STICKY`, which Mutter honours; layer-shell overlays are
        // already global. Must come after mapping.
        if !use_layer_shell() && pill.ui.borrow().cfg.all_workspaces {
            pill.window.stick();
        }
        ensure_tick(pill);
    });
}

/// Unmap the pill. Main thread only.
pub fn hide() {
    with_pill(|pill| pill.window.hide());
}

/// Swap in new geometry/placement settings. Unlike the webview pill, which has
/// to be torn down and rebuilt (a live resize ghosts on Wayland, wry #1667),
/// the cairo pill just takes the new size and re-places itself. Main thread only.
pub fn apply_config(cfg: PillOverlayConfig) {
    with_pill(|pill| {
        pill.ui.borrow_mut().cfg = cfg.clone();
        pill.window.set_size_request(cfg.w, cfg.h);
        pill.window.resize(cfg.w, cfg.h);
        place(pill);
        pill.area.queue_draw();
    });
}

/// Re-resolve and apply the pill's position — after a settings change, or a
/// "reset position" that cleared the saved one. Main thread only.
pub fn reposition() {
    with_pill(place);
}

// ── State in ────────────────────────────────────────────────────────────

/// Publish the live service status. Any thread.
pub fn set_status(app: &AppHandle, status: &str) {
    STATUS.store(
        match Mode::from_status(status) {
            Mode::Idle => MODE_IDLE,
            Mode::Listening => MODE_LISTENING,
            Mode::Processing => MODE_PROCESSING,
        },
        Ordering::Release,
    );
    // A status change can start animation from rest, so the tick has to be
    // (re)started on the main thread. The tick itself picks up the new value.
    let _ = app.run_on_main_thread(|| {
        with_pill(ensure_tick);
    });
}

/// Publish a mic level (0.0–1.0). Any thread, at audio rate.
pub fn set_level(level: f32) {
    let prev = f32::from_bits(LEVEL.load(Ordering::Acquire));
    let alpha = if level > prev {
        LEVEL_ATTACK
    } else {
        LEVEL_RELEASE
    };
    LEVEL.store((prev + alpha * (level - prev)).to_bits(), Ordering::Release);
}

// ── Placement ───────────────────────────────────────────────────────────

/// Whether to drive the pill as a Wayland layer-shell overlay: only when the
/// compositor actually implements `wlr-layer-shell`. GNOME/Mutter is Wayland but
/// ships none, and calling `init_layer_shell()` there falls back to a plain XDG
/// toplevel that silently ignores anchors and margins — breaking placement and
/// drag. `run()` puts that case on XWayland instead. Main thread only
/// (`is_supported` asserts it).
pub fn use_layer_shell() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some() && gtk_layer_shell::is_supported()
}

/// Connected monitors in logical coordinates, straight from GDK — the same
/// source tao reports through `tauri::Monitor`, including the connector name
/// used as the monitor-pin key. Main thread only.
pub fn monitors() -> Vec<MonitorRect> {
    let Some(display) = gdk::Display::default() else {
        return Vec::new();
    };
    (0..display.n_monitors())
        .filter_map(|i| display.monitor(i))
        .map(|m| {
            let g = m.geometry();
            MonitorRect {
                name: m.model().map(|s| s.to_string()),
                x: g.x(),
                y: g.y(),
                w: g.width(),
                h: g.height(),
                primary: m.is_primary(),
                // GDK geometry is already logical for the whole layout, so the
                // placement space and the pill's own units are the same here.
                scale: 1.0,
            }
        })
        .collect()
}

/// Resolve the pill's position and apply it. Runs on every show, so a monitor
/// hotplug or resolution change is picked up rather than baked in at creation.
fn place(pill: &Pill) {
    let (x, y) = crate::pill::resolve_pill_position(&monitors());
    move_to(&pill.window, x, y);
}

fn move_to(window: &gtk::Window, x: i32, y: i32) {
    if use_layer_shell() {
        window.set_layer_shell_margin(gtk_layer_shell::Edge::Left, x);
        window.set_layer_shell_margin(gtk_layer_shell::Edge::Top, y);
    } else {
        window.move_(x, y);
    }
}

// ── Animation ───────────────────────────────────────────────────────────

/// Start the frame-clock callback if it isn't already running. Stopping when
/// nothing moves is the point: an always-on tick keeps requesting frames from
/// the compositor for an overlay that is idle almost all of the time.
fn ensure_tick(pill: &Pill) {
    {
        let mut ui = pill.ui.borrow_mut();
        if ui.ticking {
            return;
        }
        ui.ticking = true;
    }
    let ui = pill.ui.clone();
    let _ = pill.area.add_tick_callback(move |area, _| {
        let keep = ui.borrow_mut().advance();
        // Draw before stopping, so the last frame of an animation is the
        // resting state rather than one frame short of it.
        area.queue_draw();
        if keep {
            glib::ControlFlow::Continue
        } else {
            ui.borrow_mut().ticking = false;
            glib::ControlFlow::Break
        }
    });
}

// ── Input ───────────────────────────────────────────────────────────────

fn connect_input(area: &gtk::DrawingArea, ui: &Rc<RefCell<Ui>>, app: &AppHandle) {
    area.connect_button_press_event({
        let ui = ui.clone();
        move |_, ev| {
            if ev.button() != PRIMARY_BUTTON {
                return glib::Propagation::Proceed;
            }
            let mut u = ui.borrow_mut();
            let frame = u.frame();
            let (lx, ly) = ev.position();
            u.press = Some(Press {
                root: ev.root(),
                local: (lx, ly),
                origin: crate::pill::current_pill_position().unwrap_or((0, 0)),
                dragging: false,
                cap: cap_at(&frame, lx, ly),
            });
            drop(u);
            with_pill(ensure_tick);
            glib::Propagation::Stop
        }
    });

    area.connect_motion_notify_event({
        let ui = ui.clone();
        move |_, ev| {
            let (lx, ly) = ev.position();
            let mut u = ui.borrow_mut();
            let frame = u.frame();
            u.hover = true;
            u.hovered_cap = cap_at(&frame, lx, ly);

            let locked = u.cfg.locked;
            let Some(press) = u.press.as_mut() else {
                drop(u);
                with_pill(ensure_tick);
                return glib::Propagation::Stop;
            };
            if locked {
                drop(u);
                with_pill(ensure_tick);
                return glib::Propagation::Stop;
            }
            let (rx, ry) = ev.root();
            if !press.dragging {
                // Below the threshold it is still a potential click, tested
                // against displacement since the press rather than this event
                // alone — a slow drag arrives as many sub-threshold events.
                if (rx - press.root.0).abs() + (ry - press.root.1).abs() < DRAG_THRESHOLD {
                    drop(u);
                    with_pill(ensure_tick);
                    return glib::Propagation::Stop;
                }
                press.dragging = true;
            }
            let target = if crate::pill::global_coords() {
                let (ox, oy) = press.origin;
                (
                    ox + (rx - press.root.0).round() as i32,
                    oy + (ry - press.root.1).round() as i32,
                )
            } else {
                let (px, py) = crate::pill::current_pill_position().unwrap_or(press.origin);
                (
                    px + (lx - press.local.0).round() as i32,
                    py + (ly - press.local.1).round() as i32,
                )
            };
            drop(u);
            let (x, y) = crate::pill::store_clamped_position(&monitors(), target.0, target.1);
            with_pill(|pill| move_to(&pill.window, x, y));
            glib::Propagation::Stop
        }
    });

    area.connect_button_release_event({
        let ui = ui.clone();
        let app = app.clone();
        move |_, ev| {
            let mut u = ui.borrow_mut();
            let Some(press) = u.press.take() else {
                return glib::Propagation::Proceed;
            };
            if press.dragging {
                drop(u);
                crate::pill::persist_pill_position();
                return glib::Propagation::Stop;
            }
            let frame = u.frame();
            let (lx, ly) = ev.position();
            // The press and the release must land on the same control, so a
            // press on the bar that drifts onto a cap doesn't fire the cap.
            let cap = press.cap.filter(|&c| cap_at(&frame, lx, ly) == Some(c));
            // The debut/dwell gate lets what's drawn lag what the runtime is
            // doing by up to a few hundred ms. A click landing in that window
            // would act on a state the user cannot see — starting a session
            // from a frame that still shows "listening", say — so it is dropped
            // rather than guessed at. The user sees nothing happen and clicks
            // again, which is the harmless outcome.
            let live = mode_from_status();
            let stale = u.mode != live;
            if cap == Some(Cap::Discard) && !stale {
                u.flash_at = Some(Instant::now());
            }
            drop(u);
            with_pill(ensure_tick);
            if stale {
                return glib::Propagation::Stop;
            }

            let runtime = app.state::<crate::runtime::AppRuntime>().inner().clone();
            tauri::async_runtime::spawn(async move {
                // The runtime debounces all of these itself (150 ms / 80 ms),
                // so the pill doesn't need a guard of its own.
                let _ = match cap {
                    Some(Cap::Discard) => runtime.cancel_dictation().await,
                    // Accept stops the session, which is what the bar does too;
                    // the chip is the explicit way to say it.
                    Some(Cap::Accept) => runtime.toggle_dictation().await.map(|_| ()),
                    // Processing is a passive read-out: finalising can't be
                    // interrupted, so the bar isn't a control there.
                    None if live == Mode::Processing => Ok(()),
                    None => runtime.toggle_dictation().await.map(|_| ()),
                };
            });
            glib::Propagation::Stop
        }
    });

    area.connect_enter_notify_event({
        let ui = ui.clone();
        move |_, _| {
            ui.borrow_mut().hover = true;
            with_pill(ensure_tick);
            glib::Propagation::Proceed
        }
    });

    area.connect_leave_notify_event({
        let ui = ui.clone();
        move |_, _| {
            let mut u = ui.borrow_mut();
            u.hover = false;
            u.hovered_cap = None;
            drop(u);
            with_pill(ensure_tick);
            glib::Propagation::Proceed
        }
    });
}

/// Run `f` against the live pill, if there is one. Main thread only.
fn with_pill<R>(f: impl FnOnce(&Pill) -> R) -> Option<R> {
    PILL.with_borrow(|slot| slot.as_ref().map(f))
}
