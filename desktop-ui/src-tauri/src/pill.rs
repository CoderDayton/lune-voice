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
use std::sync::atomic::{AtomicI32, Ordering};
use tauri::{AppHandle, Manager, WebviewWindow};

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
    PILL_CFG.lock().expect("pill config lock").clone()
}

/// Seed the live config from settings at startup, before the pill window is
/// created, so the first window is built with the user's geometry.
pub fn seed_pill_config(cfg: PillOverlayConfig) {
    *PILL_CFG.lock().expect("pill config lock") = cfg;
}

/// Live pill position (logical px from the monitor's top-left). `i32::MIN` =
/// uninitialised — `resolve_pill_position` then loads the saved value or
/// computes the default.
static PILL_X: AtomicI32 = AtomicI32::new(i32::MIN);
static PILL_Y: AtomicI32 = AtomicI32::new(i32::MIN);
/// Cached logical bounds of the monitor the pill was last placed on, for
/// clamping drags. Origin is 0 on the layer-shell path, whose margins are
/// per-output; elsewhere it is the monitor's offset in the global coordinate
/// space, so a screen stacked below another doesn't clamp the pill off-screen.
static MON_X: AtomicI32 = AtomicI32::new(0);
static MON_Y: AtomicI32 = AtomicI32::new(0);
static MON_W: AtomicI32 = AtomicI32::new(0);
static MON_H: AtomicI32 = AtomicI32::new(0);

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

/// One connected monitor in logical (scale-divided) coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonitorRect {
    /// Connector name — the key the pill's monitor pin is persisted under.
    pub name: Option<String>,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub primary: bool,
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
                let scale = if scale.is_finite() && scale > 0.0 {
                    scale
                } else {
                    1.0
                };
                let (pos, size) = (m.position(), m.size());
                let name = m.name().cloned();
                MonitorRect {
                    primary: name.is_some() && name == primary,
                    name,
                    x: (f64::from(pos.x) / scale) as i32,
                    y: (f64::from(pos.y) / scale) as i32,
                    w: (f64::from(size.width) / scale) as i32,
                    h: (f64::from(size.height) / scale) as i32,
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

/// Clamp so the whole pill stays inside `m`.
fn clamp_within(m: &MonitorRect, cfg: &PillOverlayConfig, x: i32, y: i32) -> (i32, i32) {
    (
        x.clamp(m.x, (m.x + m.w - cfg.w).max(m.x)),
        y.clamp(m.y, (m.y + m.h - cfg.h).max(m.y)),
    )
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
        return match monitor_at(mons, x + cfg.w / 2, y + cfg.h / 2) {
            Some(m) => clamp_within(m, &cfg, x, y),
            // Unreadable monitor layout — better unclamped than pinned to (0,0).
            None => (x, y),
        };
    }
    let m = MonitorRect {
        name: None,
        x: MON_X.load(Ordering::Acquire),
        y: MON_Y.load(Ordering::Acquire),
        w: MON_W.load(Ordering::Acquire),
        h: MON_H.load(Ordering::Acquire),
        primary: false,
    };
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
    }
    let target = current_pill_position().unwrap_or_else(|| {
        anchored_position(
            &pill_cfg(),
            MON_X.load(Ordering::Acquire),
            MON_Y.load(Ordering::Acquire),
            MON_W.load(Ordering::Acquire),
            MON_H.load(Ordering::Acquire),
        )
    });
    store_clamped_position(mons, target.0, target.1)
}

/// The pill's default position for an anchor + edge margin on a monitor at
/// (mx, my) sized mw×mh (logical px). Pure — unit-tested below.
fn anchored_position(cfg: &PillOverlayConfig, mx: i32, my: i32, mw: i32, mh: i32) -> (i32, i32) {
    use PillAnchor::{BottomLeft, BottomRight, TopCenter, TopLeft, TopRight};
    let m = cfg.edge_margin;
    let x = match cfg.anchor {
        TopLeft | BottomLeft => mx + m,
        TopRight | BottomRight => mx + (mw - cfg.w - m).max(0),
        // Centred anchors (and any future variant): horizontally centred.
        _ => mx + ((mw - cfg.w) / 2).max(0),
    };
    let y = match cfg.anchor {
        TopLeft | TopCenter | TopRight => my + m,
        // Bottom anchors (and any future variant): above the bottom edge.
        _ => (my + mh - cfg.h - m).max(my),
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
        if let Err(e) = WebviewWindowBuilder::new(app, "pill", WebviewUrl::App("pill.html".into()))
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
            .focused(false)
            .visible(false)
            .build()
        {
            tracing::error!(error = %e, "failed to create pill overlay window");
        }
    }
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
            let _ = pill.set_visible_on_all_workspaces(pill_cfg().all_workspaces);
            let (x, y) = resolve_pill_position(&monitors(&pill));
            let _ = pill.set_position(tauri::LogicalPosition::new(x, y));
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
        let mut g = PILL_CFG.lock().expect("pill config lock");
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
            for _ in 0..40 {
                if handle.get_webview_window("pill").is_none() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
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

/// Move the pill by a drag delta (logical px), clamped on-screen. Only the
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
    let desired = (
        PILL_X.load(Ordering::Acquire).saturating_add(dx),
        PILL_Y.load(Ordering::Acquire).saturating_add(dy),
    );
    let handle = app.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.run_on_main_thread(move || {
        if let Some(pill) = handle.get_webview_window("pill") {
            // Clamping needs the live monitor layout, hence the main thread: a
            // drag across monitors must stay on screen rather than vanish into
            // the gap between them.
            let (x, y) = store_clamped_position(&monitors(&pill), desired.0, desired.1);
            let _ = pill.set_position(tauri::LogicalPosition::new(x, y));
        }
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
            let _ = pill.set_position(tauri::LogicalPosition::new(x, y));
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
    use super::{MonitorRect, PillOverlayConfig, anchored_position, monitor_at, should_show_pill};
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
        MonitorRect {
            name: None,
            x,
            y,
            w,
            h,
            primary: false,
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
                anchored_position(&cfg(anchor, m), 0, 0, 1920, 1080),
                expected,
                "{anchor:?}"
            );
        }
    }

    #[test]
    fn anchored_position_honours_monitor_origin_and_clamps() {
        // Secondary monitor at (1920, 200): offsets apply.
        assert_eq!(
            anchored_position(&cfg(PillAnchor::TopLeft, 10), 1920, 200, 1280, 720),
            (1930, 210)
        );
        // Margin larger than the monitor never pushes the pill off-screen left/top.
        let (x, y) = anchored_position(&cfg(PillAnchor::BottomRight, 5000), 0, 0, 800, 600);
        assert!(x >= 0 && y >= 0, "clamped: ({x}, {y})");
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
