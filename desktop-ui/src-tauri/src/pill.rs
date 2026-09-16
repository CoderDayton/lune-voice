// SPDX-License-Identifier: GPL-3.0-or-later

//! The dictation pill overlay: geometry, placement and lifecycle.
//!
//! Two implementations sit behind this module. On Linux the pill is a bare GTK
//! window drawn with cairo (`pill_native` + `pill_paint`) — WebKitGTK cannot
//! composite alpha on that stack, so a webview pill could only ever be a flat
//! opaque rectangle. Everywhere else it is a Tauri webview window rendering
//! `pill.html`. Everything in *this* file — the config, the saved position and
//! the placement maths — is shared by both.

use lune_core::config::{
    PillAnchor, PillCorner, PillOrientation, PillSize, PillVisibility, UserSettings,
};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
// `emit_to` lives on this trait. Only the webview pill emits — the native
// Linux pill takes its state by direct call — so importing it there would be
// an unused import.
#[cfg(not(target_os = "linux"))]
use tauri::Emitter;
// Only the webview pill looks windows up by label; the Linux pill is a GTK
// window Tauri does not know about.
#[cfg(not(target_os = "linux"))]
use tauri::Manager;
use tauri::{AppHandle, WebviewWindow};

use crate::commands::require_window_access;

/// Live pill overlay configuration derived from user settings. The window IS
/// the pill's footprint: `w`/`h` are what gets placed, sized and drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct PillOverlayConfig {
    pub w: i32,
    pub h: i32,
    /// Corner radius (logical px).
    pub radius: i32,
    pub anchor: PillAnchor,
    /// Gap (logical px) between the pill and the screen edges in its default
    /// anchor placement, so it clears a taskbar/dock.
    pub edge_margin: i32,
    /// Monitor connector name the pill is pinned to; `None` = automatic.
    pub monitor: Option<String>,
    /// Dragging disabled.
    pub locked: bool,
    /// Show on every workspace (X11 `_NET_WM_STATE_STICKY` / macOS spaces).
    /// Windows has no equivalent — its virtual desktops are not addressable
    /// this way — so the flag is ignored there.
    pub all_workspaces: bool,
}

impl PillOverlayConfig {
    pub fn from_settings(s: &UserSettings) -> Self {
        // Footprints are deliberately generous: the pill is a glanceable
        // overlay on a busy desktop, and the previous set read as a thin strip
        // that disappeared into dark backgrounds. Each step is ~1.2× the last
        // so the three options stay visibly distinct.
        let (long, short) = match s.pill_size {
            PillSize::Compact => (140, 42),
            PillSize::Large => (200, 60),
            _ => (168, 50),
        };
        let (w, h) = match s.pill_orientation {
            PillOrientation::Vertical => (short, long),
            _ => (long, short),
        };
        let radius = match s.pill_corner {
            PillCorner::Pill => short / 2,
            PillCorner::Square => 6,
            // Rounded (non-pill, non-square): keep the radius near a third of
            // the short side so the softness scales with the bar.
            _ => match s.pill_size {
                PillSize::Compact => 13,
                PillSize::Large => 18,
                _ => 15,
            },
        };
        Self {
            w,
            h,
            radius,
            anchor: s.pill_anchor,
            edge_margin: s.pill_edge_margin.min(512) as i32,
            monitor: s.pill_monitor.clone(),
            locked: s.pill_locked,
            all_workspaces: s.pill_all_workspaces,
        }
    }

    /// Whether switching from `old` to `self` changes anything the window
    /// itself is built from. `locked` is enforced per-interaction, not baked in.
    fn changes_the_window(&self, old: &Self) -> bool {
        let mut a = self.clone();
        let mut b = old.clone();
        a.locked = false;
        b.locked = false;
        a != b
    }
}

/// Live config — seeded from settings at startup (`seed_pill_config`), swapped
/// by `apply_pill_settings` on save. Defaults mirror `UserSettings::default()`
/// so a read before seeding is still sane.
static PILL_CFG: Mutex<PillOverlayConfig> = Mutex::new(PillOverlayConfig {
    w: 168,
    h: 50,
    radius: 15,
    anchor: PillAnchor::BottomCenter,
    edge_margin: 72,
    monitor: None,
    locked: false,
    all_workspaces: true,
});

pub(crate) fn pill_cfg() -> PillOverlayConfig {
    // Recover from poisoning rather than propagating it: the release profile
    // unwinds, so a panic elsewhere must not turn the pill config into a
    // second panic on every later read. Matches `hotkey_linux.rs`.
    PILL_CFG.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Seed the live config from settings at startup, before the pill window is
/// created, so the first window is built with the user's geometry.
pub fn seed_pill_config(cfg: PillOverlayConfig) {
    *PILL_CFG.lock().unwrap_or_else(|e| e.into_inner()) = cfg;
}

/// Live pill position, in the placement space [`MonitorRect`] describes.
/// `i32::MIN` = uninitialised — `resolve_pill_position` then loads the saved
/// value or computes the default.
static PILL_X: AtomicI32 = AtomicI32::new(i32::MIN);
static PILL_Y: AtomicI32 = AtomicI32::new(i32::MIN);
/// Cached bounds of the monitor the pill was last placed on, for clamping
/// drags. Origin is 0 on the layer-shell path, whose margins are per-output;
/// elsewhere it is the monitor's offset in the global coordinate space, so a
/// screen stacked below another doesn't clamp the pill off-screen.
static MON_X: AtomicI32 = AtomicI32::new(0);
static MON_Y: AtomicI32 = AtomicI32::new(0);
static MON_W: AtomicI32 = AtomicI32::new(0);
static MON_H: AtomicI32 = AtomicI32::new(0);
/// That monitor's scale, as `f32` bits so the cache stays lock-free. Zero
/// (the pre-resolve value) reads back as 1.0.
static MON_SCALE: AtomicU32 = AtomicU32::new(0);

/// The monitor the pill was last placed on, rebuilt from the cache above.
fn cached_monitor() -> MonitorRect {
    let scale = f64::from(f32::from_bits(MON_SCALE.load(Ordering::Acquire)));
    MonitorRect {
        name: None,
        x: MON_X.load(Ordering::Acquire),
        y: MON_Y.load(Ordering::Acquire),
        w: MON_W.load(Ordering::Acquire),
        h: MON_H.load(Ordering::Acquire),
        primary: false,
        scale: if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        },
    }
}

/// Seed the saved pill position at startup (from settings) so the pill reopens
/// where the user last dragged it, ahead of the first show.
pub fn set_saved_position(x: i32, y: i32) {
    PILL_X.store(x, Ordering::Release);
    PILL_Y.store(y, Ordering::Release);
}

/// The live pill position, or `None` if it was never initialised.
pub(crate) fn current_pill_position() -> Option<(i32, i32)> {
    let x = PILL_X.load(Ordering::Acquire);
    let y = PILL_Y.load(Ordering::Acquire);
    (x != i32::MIN && y != i32::MIN).then_some((x, y))
}

// ── Monitors ────────────────────────────────────────────────────────────

/// One connected monitor, in the coordinate space the pill is placed in.
///
/// A desktop spanning monitors at different DPI has exactly one coherent
/// coordinate space, and it is the device-pixel one: dividing each monitor by
/// its own scale makes neighbouring screens claim overlapping coordinates, so
/// a drag lands on the wrong one. These are therefore device pixels off the
/// platform on Windows and macOS. Linux reads GDK, which reports logical
/// coordinates for the whole layout, so `scale` is 1.0 there and the two
/// spaces coincide.
///
/// `scale` converts the pill's logical footprint into this space —
/// see [`pill_extent`].
#[derive(Clone, Debug, PartialEq)]
pub struct MonitorRect {
    /// Connector name — the key the pill's monitor pin is persisted under.
    pub name: Option<String>,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub primary: bool,
    /// Device pixels per logical pixel on this monitor.
    pub scale: f64,
}

/// Connected monitors. On Linux this reads GDK directly (the pill is a GTK
/// window and there may be no Tauri window to ask); elsewhere it goes through
/// Tauri. Both end up at the same platform data. Main thread.
fn monitors(window: &WebviewWindow) -> Vec<MonitorRect> {
    #[cfg(target_os = "linux")]
    {
        let _ = window;
        crate::pill_native::monitors()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let primary = window
            .primary_monitor()
            .ok()
            .flatten()
            .and_then(|m| m.name().cloned());
        window
            .available_monitors()
            .unwrap_or_default()
            .into_iter()
            .map(|m| {
                let scale = m.scale_factor();
                let (pos, size) = (m.position(), m.size());
                let name = m.name().cloned();
                MonitorRect {
                    primary: name.is_some() && name == primary,
                    name,
                    x: pos.x,
                    y: pos.y,
                    w: i32::try_from(size.width).unwrap_or(i32::MAX),
                    h: i32::try_from(size.height).unwrap_or(i32::MAX),
                    scale: if scale.is_finite() && scale > 0.0 {
                        scale
                    } else {
                        1.0
                    },
                }
            })
            .collect()
    }
}

/// The monitor the pill belongs on: the pinned one while it is connected, else
/// the primary, else whatever is first.
fn pill_monitor(mons: &[MonitorRect]) -> Option<&MonitorRect> {
    pill_cfg()
        .monitor
        .and_then(|pin| mons.iter().find(|m| m.name.as_deref() == Some(&pin)))
        .or_else(|| mons.iter().find(|m| m.primary))
        .or_else(|| mons.first())
}

/// The monitor containing (cx, cy), or the nearest one by centre distance.
fn monitor_at(mons: &[MonitorRect], cx: i32, cy: i32) -> Option<&MonitorRect> {
    mons.iter()
        .find(|m| cx >= m.x && cx < m.x + m.w && cy >= m.y && cy < m.y + m.h)
        .or_else(|| {
            mons.iter().min_by_key(|m| {
                let (dx, dy) = (cx - (m.x + m.w / 2), cy - (m.y + m.h / 2));
                i64::from(dx).pow(2) + i64::from(dy).pow(2)
            })
        })
}

/// The pill's footprint and edge margin in `m`'s coordinate space.
///
/// [`PillOverlayConfig`] is in logical px, because that is what the webview
/// lays itself out in and what the window is built with. Placement happens in
/// the monitor's space, which is device pixels wherever the two differ, so a
/// 168×50 pill stays 168×50 to the eye on a 150% panel instead of shrinking to
/// two thirds of its size.
fn pill_extent(cfg: &PillOverlayConfig, m: &MonitorRect) -> (i32, i32, i32) {
    let to_device = |v: i32| (f64::from(v) * m.scale).round() as i32;
    (
        to_device(cfg.w).max(1),
        to_device(cfg.h).max(1),
        to_device(cfg.edge_margin).max(0),
    )
}

/// Clamp so the whole pill stays inside `m`.
fn clamp_within(m: &MonitorRect, cfg: &PillOverlayConfig, x: i32, y: i32) -> (i32, i32) {
    let (w, h, _) = pill_extent(cfg, m);
    (
        x.clamp(m.x, (m.x + m.w - w).max(m.x)),
        y.clamp(m.y, (m.y + m.h - h).max(m.y)),
    )
}

/// The monitor a pill whose top-left is (x, y) belongs to: the one under its
/// centre.
///
/// Two passes, because the pill's size in device pixels depends on the monitor
/// being looked for. The corner picks a candidate, that candidate's scale gives
/// the footprint, and the centre picks the answer.
fn monitor_for_pill<'a>(
    mons: &'a [MonitorRect],
    cfg: &PillOverlayConfig,
    x: i32,
    y: i32,
) -> Option<&'a MonitorRect> {
    let corner = monitor_at(mons, x, y)?;
    let (w, h, _) = pill_extent(cfg, corner);
    monitor_at(mons, x + w / 2, y + h / 2)
}

/// Whether pill positions are in the compositor's global coordinate space
/// (X11/XWayland, Windows, macOS), as opposed to per-output layer-shell
/// margins (COSMIC/wlroots). The Linux check touches GTK, so main thread only.
pub(crate) fn global_coords() -> bool {
    #[cfg(target_os = "linux")]
    {
        !crate::pill_native::use_layer_shell()
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// Clamp a desired pill position on-screen. In the global coordinate space the
/// live monitor layout is used, so a drag crosses between monitors cleanly
/// instead of vanishing into the gap; on the layer-shell path margins are
/// per-output, so the cached bounds of that output apply. Main thread.
fn clamp_pill_target(mons: &[MonitorRect], x: i32, y: i32) -> (i32, i32) {
    let cfg = pill_cfg();
    if global_coords() {
        return match monitor_for_pill(mons, &cfg, x, y) {
            Some(m) => clamp_within(m, &cfg, x, y),
            // Unreadable monitor layout — better unclamped than pinned to (0,0).
            None => (x, y),
        };
    }
    let m = cached_monitor();
    if m.w <= 0 || m.h <= 0 {
        return (x, y);
    }
    clamp_within(&m, &cfg, x, y)
}

/// Clamp a position, store it as the live one, and hand it back. The one way a
/// drag updates the pill's position. Main thread.
pub(crate) fn store_clamped_position(mons: &[MonitorRect], x: i32, y: i32) -> (i32, i32) {
    let (x, y) = clamp_pill_target(mons, x, y);
    PILL_X.store(x, Ordering::Release);
    PILL_Y.store(y, Ordering::Release);
    (x, y)
}

/// Resolve the pill's position: the saved/live value if set, else the default
/// anchor placement. Caches the monitor bounds for later drag clamping and
/// stores the result. Main thread.
pub(crate) fn resolve_pill_position(mons: &[MonitorRect]) -> (i32, i32) {
    if let Some(m) = pill_monitor(mons) {
        // Layer-shell margins are per-output (origin 0); global coords keep the
        // monitor's offset so multi-monitor placement lands on the right screen.
        let (ox, oy) = if global_coords() { (m.x, m.y) } else { (0, 0) };
        MON_X.store(ox, Ordering::Release);
        MON_Y.store(oy, Ordering::Release);
        MON_W.store(m.w, Ordering::Release);
        MON_H.store(m.h, Ordering::Release);
        MON_SCALE.store((m.scale as f32).to_bits(), Ordering::Release);
    }
    let target = current_pill_position()
        .unwrap_or_else(|| anchored_position(&pill_cfg(), &cached_monitor()));
    store_clamped_position(mons, target.0, target.1)
}

/// The pill's default position for an anchor + edge margin on `m`. Pure —
/// unit-tested below.
fn anchored_position(cfg: &PillOverlayConfig, m: &MonitorRect) -> (i32, i32) {
    use PillAnchor::{BottomLeft, BottomRight, TopCenter, TopLeft, TopRight};
    let (pw, ph, margin) = pill_extent(cfg, m);
    let x = match cfg.anchor {
        TopLeft | BottomLeft => m.x + margin,
        TopRight | BottomRight => m.x + (m.w - pw - margin).max(0),
        // Centred anchors (and any future variant): horizontally centred.
        _ => m.x + ((m.w - pw) / 2).max(0),
    };
    let y = match cfg.anchor {
        TopLeft | TopCenter | TopRight => m.y + margin,
        // Bottom anchors (and any future variant): above the bottom edge.
        _ => (m.y + m.h - ph - margin).max(m.y),
    };
    (x, y)
}

/// Persist the pill's live position to settings so it reopens there. Called on
/// pointer-up after a drag; runs off the main thread.
pub(crate) fn persist_pill_position() {
    let Some((x, y)) = current_pill_position() else {
        return;
    };
    tauri::async_runtime::spawn_blocking(move || {
        if let Err(e) = write_pill_position(Some(lune_core::config::PillPosition { x, y })) {
            tracing::warn!(error = %e, "failed to save pill position");
        }
    });
}

fn write_pill_position(pos: Option<lune_core::config::PillPosition>) -> Result<(), String> {
    let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
    let store = lune_persist::SettingsStore::new(paths.settings_file());
    let mut settings = store.load().map_err(|e| format!("load settings: {e}"))?;
    settings.pill_position = pos;
    store
        .save(&settings)
        .map_err(|e| format!("save settings: {e}"))
}

// ── Visibility ──────────────────────────────────────────────────────────

/// Single source of truth for whether the pill should be on screen, given the
/// live service status and the visibility setting. Called only from Rust
/// (startup + `publish_status`) — the frontend never decides visibility, so the
/// mapped window and its content can't desync.
pub(crate) fn should_show_pill(vis: PillVisibility, status: &str) -> bool {
    let dictating = matches!(status, "listening" | "processing");
    match vis {
        PillVisibility::WhenDictating => dictating,
        // Always: visible whenever the service is up (idle or dictating).
        PillVisibility::Always => dictating || status == "ready",
        // Never + any future variant: no pill.
        _ => false,
    }
}

// ── State delivery ──────────────────────────────────────────────────────

/// Push the live service status into the pill. The native pill takes it by
/// direct call; the webview one listens for the event. Any thread.
pub fn publish_status(app: &AppHandle, status: &str) {
    #[cfg(target_os = "linux")]
    crate::pill_native::set_status(app, status);
    #[cfg(not(target_os = "linux"))]
    if let Err(e) = app.emit_to("pill", "lune://pill-state", status.to_owned()) {
        tracing::debug!(error = %e, "pill state emit skipped (pill window may not exist)");
    }
}

/// Push a mic level (0.0–1.0) into the pill. Any thread, at audio rate.
pub fn publish_audio_level(app: &AppHandle, level: f32) {
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        crate::pill_native::set_level(level);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = app.emit_to("pill", "lune://audio-level", level);
    }
}

// ── Lifecycle ───────────────────────────────────────────────────────────

/// Create the pill overlay window (hidden). Idempotent. **Must run on the main
/// thread**; at startup `setup` already is, otherwise go through
/// `set_pill_enabled`.
pub fn create_pill_window(app: &AppHandle) {
    #[cfg(target_os = "linux")]
    crate::pill_native::create(app);

    #[cfg(not(target_os = "linux"))]
    {
        use tauri::{WebviewUrl, WebviewWindowBuilder};
        if app.get_webview_window("pill").is_some() {
            return;
        }
        let cfg = pill_cfg();
        // Hand the webview its exact geometry before any page script runs, so
        // the CSS footprint matches the window from the first paint (no async
        // round-trip, no flash at the default size).
        let geom_script = format!(
            "window.__PILL_GEOM={{w:{},h:{},r:{},vertical:{}}};",
            cfg.w,
            cfg.h,
            cfg.radius,
            cfg.h > cfg.w
        );
        match WebviewWindowBuilder::new(app, "pill", WebviewUrl::App("pill.html".into()))
            .title("")
            .initialization_script(&geom_script)
            .inner_size(f64::from(cfg.w), f64::from(cfg.h))
            .resizable(false)
            .decorations(false)
            // Transparent so the bar's rounded corners can show — an opaque
            // window is a hard rectangle.
            .transparent(true)
            .always_on_top(true)
            .skip_taskbar(true)
            // Creation-time only; clicks are handled by `deny_activation`.
            .focused(false)
            .visible(false)
            .build()
        {
            Ok(_pill) => {
                #[cfg(target_os = "windows")]
                deny_activation(&_pill);
            }
            Err(e) => tracing::error!(error = %e, "failed to create pill overlay window"),
        }
    }
}

/// Stop the pill from taking focus when it is clicked.
///
/// The window holding focus when a dictation ends is the window the transcript
/// is typed into, so an overlay that activates on click eats the user's text.
/// The Linux pill refuses focus at the GTK level (`set_accept_focus(false)` in
/// `pill_native`); `focused(false)` is the builder's equivalent but covers
/// creation only, and `WS_EX_NOACTIVATE` is what covers the click.
#[cfg(target_os = "windows")]
fn deny_activation(pill: &WebviewWindow) {
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GWL_EXSTYLE, GetWindowLongPtrW, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE,
        SWP_NOZORDER, SetWindowLongPtrW, SetWindowPos, WS_EX_NOACTIVATE,
    };

    let hwnd: HWND = match pill.hwnd() {
        Ok(h) => h.0,
        Err(e) => {
            tracing::error!(
                error = %e,
                "pill window exposes no HWND; it will take focus when clicked and swallow a transcript"
            );
            return;
        }
    };
    // SAFETY: `hwnd` is this process's own live overlay window, and both calls
    // read or write window styles on it and nothing else.
    unsafe {
        let style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE as isize);
        // An extended style only takes effect once the frame is recalculated.
        SetWindowPos(
            hwnd,
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        );
    }
}

/// Move the pill window to (x, y) in the monitor coordinate space.
#[cfg(not(target_os = "linux"))]
fn place(pill: &WebviewWindow, x: i32, y: i32) {
    // Physical, not logical: `MonitorRect` is in device pixels here, and a
    // logical position would be converted using whichever monitor the window
    // currently sits on — so dragging across a DPI boundary would teleport it.
    let _ = pill.set_position(tauri::PhysicalPosition::new(x, y));
}

/// Leftover fraction of a device pixel from the last scaled drag delta, one
/// axis each, as `f32` bits so the carry stays lock-free.
#[cfg_attr(target_os = "linux", allow(dead_code))]
static DRAG_REM_X: AtomicU32 = AtomicU32::new(0);
#[cfg_attr(target_os = "linux", allow(dead_code))]
static DRAG_REM_Y: AtomicU32 = AtomicU32::new(0);

/// Scale one axis of a drag delta into device pixels, carrying the fraction
/// that did not fit into `rem` for the next frame.
///
/// Deltas arrive one animation frame at a time, so rounding each on its own
/// compounds: a slow 1 CSS px/frame drag on a 150% panel would round to 2
/// device px every frame and the pill would run away from the cursor. The
/// carry makes the drag exact over its whole length. Portable — unit-tested
/// below.
#[cfg_attr(target_os = "linux", allow(dead_code))]
fn scaled_with_carry(v: i32, scale: f64, rem: &AtomicU32) -> i32 {
    let carried = f64::from(f32::from_bits(rem.load(Ordering::Acquire)));
    let exact = f64::from(v) * scale + if carried.is_finite() { carried } else { 0.0 };
    let whole = exact.trunc();
    rem.store(((exact - whole) as f32).to_bits(), Ordering::Release);
    whole as i32
}

/// Convert a webview pointer delta (CSS px) into the placement space.
#[cfg(not(target_os = "linux"))]
fn scale_delta(pill: &WebviewWindow, dx: i32, dy: i32) -> (i32, i32) {
    let scale = pill.scale_factor().unwrap_or(1.0);
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    (
        scaled_with_carry(dx, scale, &DRAG_REM_X),
        scaled_with_carry(dy, scale, &DRAG_REM_Y),
    )
}

/// Wait for a destroyed window's label to come free, up to 5 seconds.
///
/// `destroy` unregisters on the event loop, so the label outlives the call.
/// Building under one that is still taken is a silent no-op.
#[cfg(not(target_os = "linux"))]
async fn label_released(app: &AppHandle, label: &str) -> bool {
    const INTERVAL: std::time::Duration = std::time::Duration::from_millis(25);
    const ATTEMPTS: u32 = 200;
    for _ in 0..ATTEMPTS {
        if app.get_webview_window(label).is_none() {
            return true;
        }
        tokio::time::sleep(INTERVAL).await;
    }
    app.get_webview_window(label).is_none()
}

/// Destroy the pill overlay window. Used when the user sets pill visibility to
/// Never — on the webview path hiding alone keeps a whole WebKitWebProcess
/// (~200 MB) alive. **Must run on the main thread**; go through
/// `set_pill_enabled` off it.
pub fn destroy_pill_window(app: &AppHandle) {
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        crate::pill_native::destroy();
    }
    #[cfg(not(target_os = "linux"))]
    if let Some(pill) = app.get_webview_window("pill")
        && let Err(e) = pill.close()
    {
        tracing::warn!(error = %e, "failed to close pill overlay window");
    }
}

/// Map the pill directly from Rust — bypasses the frontend IPC round trip so
/// the hotkey response appears instant. Any thread.
pub fn show_pill_direct(app: &AppHandle) {
    on_main_thread(app, |_app| {
        #[cfg(target_os = "linux")]
        crate::pill_native::show();

        #[cfg(not(target_os = "linux"))]
        if let Some(pill) = _app.get_webview_window("pill") {
            // TODO(macos): use an NSPanel with floatingPanel +
            // NSNonactivatingPanelMask for proper overlay behaviour; plain
            // always_on_top shows in Mission Control.
            // tao leaves this unimplemented on Windows, whose virtual desktops
            // are not addressable per-window.
            #[cfg(not(target_os = "windows"))]
            let _ = pill.set_visible_on_all_workspaces(pill_cfg().all_workspaces);
            let (x, y) = resolve_pill_position(&monitors(&pill));
            place(&pill, x, y);
            let _ = pill.show();
        }
    });
}

/// Unmap the pill. Any thread.
pub fn hide_pill_direct(app: &AppHandle) {
    on_main_thread(app, |_app| {
        #[cfg(target_os = "linux")]
        crate::pill_native::hide();

        #[cfg(not(target_os = "linux"))]
        if let Some(pill) = _app.get_webview_window("pill") {
            let _ = pill.hide();
        }
    });
}

/// Reconcile the pill's existence with whether it is enabled (visibility !=
/// Never), from any thread.
pub fn set_pill_enabled(app: &AppHandle, enabled: bool) {
    on_main_thread(app, move |app| {
        if enabled {
            create_pill_window(app);
        } else {
            destroy_pill_window(app);
        }
    });
}

/// Push freshly-saved pill settings into the live overlay. `show_after` is the
/// caller's visibility decision (from `should_show_pill`). Any thread.
pub fn apply_pill_settings(app: &AppHandle, settings: &UserSettings, show_after: bool) {
    let new = PillOverlayConfig::from_settings(settings);
    let saved = settings.pill_position;
    let old = {
        let mut g = PILL_CFG.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::replace(&mut *g, new.clone())
    };
    // Never = no window to update; set_pill_enabled destroys it separately and
    // a later re-enable builds it from the (already swapped) config.
    if !new.changes_the_window(&old) || settings.pill_visibility == PillVisibility::Never {
        return;
    }
    // Re-seed the live position from the (possibly just-cleared) saved one, so
    // placement resolves fresh instead of inheriting the old spot.
    let (x, y) = saved.map_or((i32::MIN, i32::MIN), |p| (p.x, p.y));
    PILL_X.store(x, Ordering::Release);
    PILL_Y.store(y, Ordering::Release);

    #[cfg(target_os = "linux")]
    {
        let _ = show_after; // the native pill is never torn down, so it can't lose its mapping
        on_main_thread(app, move |_| crate::pill_native::apply_config(new));
    }

    #[cfg(not(target_os = "linux"))]
    on_main_thread(app, move |app| {
        let Some(pill) = app.get_webview_window("pill") else {
            return; // not created — nothing stale to rebuild
        };
        // The window is never resized live (compositor ghost artifacts during
        // resize, wry #1667, tauri #14924) — a geometry change rebuilds it.
        // destroy() unregisters via the event loop, so recreating under the
        // same label has to wait for it to land.
        if let Err(e) = pill.destroy() {
            tracing::warn!(error = %e, "failed to destroy pill window for rebuild");
            return;
        }
        let handle = app.clone();
        tauri::async_runtime::spawn(async move {
            if !label_released(&handle, "pill").await {
                tracing::error!(
                    "pill window label still registered 5s after destroy; the overlay stays \
                     hidden until the next pill settings change or a restart"
                );
                return;
            }
            let inner = handle.clone();
            let _ = handle.run_on_main_thread(move || {
                create_pill_window(&inner);
                if show_after {
                    show_pill_direct(&inner);
                }
            });
        });
    });
}

fn on_main_thread(app: &AppHandle, task: impl FnOnce(&AppHandle) + Send + 'static) {
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || task(&handle));
}

// ── Commands ────────────────────────────────────────────────────────────

/// Move the pill by a drag delta in the webview's CSS pixels, clamped
/// on-screen. Only the
/// webview pill goes through here — the native one moves its own GTK window
/// from the pointer events it already receives, so on Linux there is no window
/// under this label and the command is inert.
#[tauri::command]
pub async fn nudge_pill(
    dx: i32,
    dy: i32,
    window: WebviewWindow,
    app: AppHandle,
) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    if pill_cfg().locked {
        return Ok(()); // position locked — backend guard behind the UI's own check
    }
    let handle = app.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.run_on_main_thread(move || {
        #[cfg(not(target_os = "linux"))]
        if let Some(pill) = handle.get_webview_window("pill") {
            // Pointer deltas arrive in the webview's CSS pixels, which is the
            // pill's own logical space; placement is in the monitor's.
            let (dx, dy) = scale_delta(&pill, dx, dy);
            let desired = (
                PILL_X.load(Ordering::Acquire).saturating_add(dx),
                PILL_Y.load(Ordering::Acquire).saturating_add(dy),
            );
            // Clamping needs the live monitor layout, hence the main thread: a
            // drag across monitors must stay on screen rather than vanish into
            // the gap between them.
            let (x, y) = store_clamped_position(&monitors(&pill), desired.0, desired.1);
            place(&pill, x, y);
        }
        #[cfg(target_os = "linux")]
        let _ = (&handle, dx, dy);
        let _ = tx.send(());
    })
    .map_err(|e| e.to_string())?;
    rx.await.map_err(|_| "nudge pill did not complete".into())
}

/// Persist the pill's resting position after a drag.
#[tauri::command]
pub async fn save_pill_position(window: WebviewWindow) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    persist_pill_position();
    Ok(())
}

/// Clear the pill's saved position and re-place it at its default anchor.
/// Invoked from the settings UI's "Reset position" button.
#[tauri::command]
pub async fn reset_pill_position(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    tauri::async_runtime::spawn_blocking(|| write_pill_position(None))
        .await
        .map_err(|e| format!("reset pill position task: {e}"))??;
    on_main_thread(&app, |_app| {
        PILL_X.store(i32::MIN, Ordering::Release);
        PILL_Y.store(i32::MIN, Ordering::Release);
        #[cfg(target_os = "linux")]
        crate::pill_native::reposition();

        #[cfg(not(target_os = "linux"))]
        if let Some(pill) = _app.get_webview_window("pill") {
            let (x, y) = resolve_pill_position(&monitors(&pill));
            place(&pill, x, y);
        }
    });
    Ok(())
}

/// Connected monitors, for the pill's monitor picker. Only named monitors are
/// listed — the connector name is the persisted pin key.
#[tauri::command]
pub async fn list_monitors(
    window: WebviewWindow,
) -> Result<Vec<crate::dto::MonitorInfoDto>, String> {
    require_window_access(&window, &["main"])?;
    let win = window.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    window
        .run_on_main_thread(move || {
            let listed = monitors(&win)
                .into_iter()
                .filter_map(|m| {
                    Some(crate::dto::MonitorInfoDto {
                        is_primary: m.primary,
                        width: m.w.max(0) as u32,
                        height: m.h.max(0) as u32,
                        name: m.name?,
                    })
                })
                .collect();
            let _ = tx.send(listed);
        })
        .map_err(|e| e.to_string())?;
    rx.await
        .map_err(|_| "list monitors did not complete".into())
}

#[cfg(test)]
mod tests {
    use super::{
        AtomicU32, MonitorRect, PillOverlayConfig, anchored_position, clamp_within, monitor_at,
        monitor_for_pill, scaled_with_carry, should_show_pill,
    };
    use lune_core::config::PillVisibility::{Always, Never, WhenDictating};
    use lune_core::config::{PillAnchor, PillCorner, PillOrientation, PillSize, UserSettings};

    fn cfg(anchor: PillAnchor, margin: i32) -> PillOverlayConfig {
        PillOverlayConfig {
            w: 132,
            h: 40,
            radius: 12,
            anchor,
            edge_margin: margin,
            monitor: None,
            locked: false,
            all_workspaces: true,
        }
    }

    fn monitor(x: i32, y: i32, w: i32, h: i32) -> MonitorRect {
        scaled_monitor(x, y, w, h, 1.0)
    }

    fn scaled_monitor(x: i32, y: i32, w: i32, h: i32, scale: f64) -> MonitorRect {
        MonitorRect {
            name: None,
            x,
            y,
            w,
            h,
            primary: false,
            scale,
        }
    }

    #[test]
    fn anchored_position_covers_all_presets() {
        // 1920×1080 monitor at (0, 0), 72px margin, 132×40 pill.
        let m = 72;
        let cases = [
            (PillAnchor::TopLeft, (72, 72)),
            (PillAnchor::TopCenter, ((1920 - 132) / 2, 72)),
            (PillAnchor::TopRight, (1920 - 132 - 72, 72)),
            (PillAnchor::BottomLeft, (72, 1080 - 40 - 72)),
            (PillAnchor::BottomCenter, ((1920 - 132) / 2, 1080 - 40 - 72)),
            (PillAnchor::BottomRight, (1920 - 132 - 72, 1080 - 40 - 72)),
        ];
        for (anchor, expected) in cases {
            assert_eq!(
                anchored_position(&cfg(anchor, m), &monitor(0, 0, 1920, 1080)),
                expected,
                "{anchor:?}"
            );
        }
    }

    #[test]
    fn anchored_position_honours_monitor_origin_and_clamps() {
        // Secondary monitor at (1920, 200): offsets apply.
        assert_eq!(
            anchored_position(
                &cfg(PillAnchor::TopLeft, 10),
                &monitor(1920, 200, 1280, 720)
            ),
            (1930, 210)
        );
        // Margin larger than the monitor never pushes the pill off-screen left/top.
        let (x, y) = anchored_position(
            &cfg(PillAnchor::BottomRight, 5000),
            &monitor(0, 0, 800, 600),
        );
        assert!(x >= 0 && y >= 0, "clamped: ({x}, {y})");
    }

    /// A 2560×1440 panel at 150% to the right of a 1920×1080 at 100% — the
    /// laptop-plus-external case. Both live in one device-pixel space, so the
    /// 132×40 pill occupies 198×60 on the scaled panel and its 72px margin is
    /// 108.
    fn mixed_dpi_pair() -> [MonitorRect; 2] {
        [
            scaled_monitor(0, 0, 1920, 1080, 1.0),
            scaled_monitor(1920, 0, 2560, 1440, 1.5),
        ]
    }

    #[test]
    fn anchored_position_scales_the_pill_to_the_monitors_dpi() {
        let [plain, scaled] = mixed_dpi_pair();
        let c = cfg(PillAnchor::BottomRight, 72);
        assert_eq!(
            anchored_position(&c, &scaled),
            (1920 + 2560 - 198 - 108, 1440 - 60 - 108)
        );
        // The unscaled panel keeps its logical placement — same pill, same
        // numbers as before any of this existed.
        assert_eq!(
            anchored_position(&c, &plain),
            (1920 - 132 - 72, 1080 - 40 - 72)
        );
    }

    #[test]
    fn a_pill_dragged_off_a_scaled_monitor_clamps_to_its_device_pixels() {
        let [plain, scaled] = mixed_dpi_pair();
        let c = cfg(PillAnchor::BottomRight, 72);
        // Dragged past the right edge: the whole 198px-wide pill stays on
        // screen, not the 132px logical one.
        assert_eq!(clamp_within(&scaled, &c, 9000, 0).0, 1920 + 2560 - 198);
        assert_eq!(clamp_within(&scaled, &c, 0, 9000).1, 1440 - 60);
        // And on the 100% panel the footprint is the logical size.
        assert_eq!(clamp_within(&plain, &c, 9000, 0).0, 1920 - 132);
    }

    #[test]
    fn the_pill_belongs_to_the_screen_its_scaled_centre_is_over() {
        // The scaled panel on the left this time, the laptop to its right.
        let mons = [
            scaled_monitor(0, 0, 2560, 1440, 1.5),
            scaled_monitor(2560, 0, 1920, 1080, 1.0),
        ];
        let c = cfg(PillAnchor::BottomCenter, 72);
        // The top-left is still on the scaled panel, where the pill is 198
        // wide, so its centre at 2579 has crossed onto the laptop. At the
        // logical 132 it would not have.
        assert_eq!(monitor_for_pill(&mons, &c, 2480, 100), Some(&mons[1]));
        // Well inside the scaled panel.
        assert_eq!(monitor_for_pill(&mons, &c, 100, 100), Some(&mons[0]));
        // No monitors at all: nothing to belong to, and the caller leaves the
        // drag unclamped rather than pinning it to the origin.
        assert_eq!(monitor_for_pill(&[], &c, 0, 0), None);
    }

    #[test]
    fn a_slow_drag_on_a_fractional_scale_keeps_its_length() {
        let rem = AtomicU32::new(0);
        // Forty frames of one CSS pixel each on a 150% panel is 60 device
        // pixels. Rounding each frame on its own would give 80 and the pill
        // would outrun the cursor by a third.
        let out: i32 = (0..40).map(|_| scaled_with_carry(1, 1.5, &rem)).sum();
        assert_eq!(out, 60);
        // Dragging back retraces exactly the same path.
        let back: i32 = (0..40).map(|_| scaled_with_carry(-1, 1.5, &rem)).sum();
        assert_eq!(back, -60);
        // An unscaled panel is the identity, carry or no carry.
        let plain = AtomicU32::new(0);
        assert_eq!(scaled_with_carry(7, 1.0, &plain), 7);
        assert_eq!(scaled_with_carry(-7, 1.0, &plain), -7);
    }

    #[test]
    fn a_dragged_pill_picks_the_monitor_its_centre_is_over() {
        let mons = [monitor(0, 0, 1920, 1080), monitor(1920, 0, 1280, 720)];
        assert_eq!(monitor_at(&mons, 100, 100), Some(&mons[0]));
        assert_eq!(monitor_at(&mons, 2000, 100), Some(&mons[1]));
        // In the gap below the shorter screen: the nearest one wins rather than
        // the pill vanishing between them.
        assert_eq!(monitor_at(&mons, 2500, 900), Some(&mons[1]));
        assert_eq!(monitor_at(&[], 0, 0), None);
    }

    #[test]
    fn overlay_config_geometry_matches_size_orientation_and_corner() {
        let mut s = UserSettings::default();
        // Standard horizontal, rounded corners — the shipped default.
        let d = PillOverlayConfig::from_settings(&s);
        assert_eq!((d.w, d.h, d.radius), (168, 50, 15));
        assert_eq!(d.edge_margin, 72);

        s.pill_size = PillSize::Compact;
        s.pill_orientation = PillOrientation::Vertical;
        s.pill_corner = PillCorner::Pill;
        let c = PillOverlayConfig::from_settings(&s);
        // Vertical swaps the axes; pill corners use half the short side.
        assert_eq!((c.w, c.h, c.radius), (42, 140, 21));

        s.pill_size = PillSize::Large;
        s.pill_orientation = PillOrientation::Horizontal;
        s.pill_corner = PillCorner::Square;
        let l = PillOverlayConfig::from_settings(&s);
        assert_eq!((l.w, l.h, l.radius), (200, 60, 6));
    }

    #[test]
    fn lock_only_changes_leave_the_window_alone() {
        let a = cfg(PillAnchor::BottomCenter, 72);
        let mut b = a.clone();
        b.locked = true;
        assert!(!b.changes_the_window(&a));
        b.edge_margin = 100;
        assert!(b.changes_the_window(&a));
    }

    #[test]
    fn always_shows_whenever_the_service_is_up() {
        // Includes idle ("ready") — the case the desync bug left invisible.
        assert!(should_show_pill(Always, "ready"));
        assert!(should_show_pill(Always, "listening"));
        assert!(should_show_pill(Always, "processing"));
        // But not while down/erroring.
        assert!(!should_show_pill(Always, "error"));
        assert!(!should_show_pill(Always, "initializing"));
    }

    #[test]
    fn when_dictating_shows_only_during_a_session() {
        assert!(should_show_pill(WhenDictating, "listening"));
        assert!(should_show_pill(WhenDictating, "processing"));
        assert!(!should_show_pill(WhenDictating, "ready"));
        assert!(!should_show_pill(WhenDictating, "error"));
    }

    #[test]
    fn never_is_always_hidden() {
        for status in ["ready", "listening", "processing", "error"] {
            assert!(!should_show_pill(Never, status));
        }
    }
}
