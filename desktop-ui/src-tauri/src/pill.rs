use lune_core::config::PillVisibility;
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

use crate::commands::require_window_access;
use std::sync::atomic::{AtomicI32, Ordering};

/// Fixed pill overlay footprint — a wide, short rounded bar. Fixed for all modes
/// to avoid compositor ghost artifacts on Wayland windows during resize (wry
/// #1667, tauri #14924). The window IS the pill's footprint; the webview content
/// fills it edge-to-edge (`.pill-root` is a fixed 40px tall), and the layer-shell
/// surface height follows that intrinsic content height, so PILL_H must match it.
const PILL_W: f64 = 132.0;
const PILL_H: f64 = 40.0;

/// Corner radius of the pill bar (logical px) — must match `.pill-root`'s
/// `border-radius` in components.css. Used to clip the X11 window shape to the
/// rounded bar (see `configure_linux_pill`).
const PILL_RADIUS: i32 = 12;

/// Gap (logical px) left below the pill in its default bottom-centre placement,
/// so it clears a bottom taskbar/dock instead of sitting flush to the edge.
const PILL_DEFAULT_BOTTOM_GAP: i32 = 72;

/// Live pill position (logical px from the monitor's top-left). `i32::MIN` =
/// uninitialised — `resolve_pill_position` then loads the saved value or
/// computes the default. Applied on Wayland via layer-shell margins, elsewhere
/// via `set_position`.
static PILL_X: AtomicI32 = AtomicI32::new(i32::MIN);
static PILL_Y: AtomicI32 = AtomicI32::new(i32::MIN);
/// Cached logical monitor size, for clamping drags on-screen (0 = unknown).
static MON_W: AtomicI32 = AtomicI32::new(0);
static MON_H: AtomicI32 = AtomicI32::new(0);
/// Cached logical origin of that monitor in the global coordinate space (0 on
/// the layer-shell path, whose margins are per-output). Lets placement/clamping
/// honour monitors offset from (0, 0) — e.g. a primary screen stacked below a
/// secondary one, where clamping to (0, 0) would push the pill off-screen.
static MON_X: AtomicI32 = AtomicI32::new(0);
static MON_Y: AtomicI32 = AtomicI32::new(0);

/// Seed the saved pill position at startup (from settings) so the pill reopens
/// where the user last dragged it, ahead of the first show/configure.
pub fn set_saved_position(x: i32, y: i32) {
    PILL_X.store(x, Ordering::Release);
    PILL_Y.store(y, Ordering::Release);
}

/// The live pill position, or `None` if it was never initialised.
pub fn current_pill_position() -> Option<(i32, i32)> {
    let x = PILL_X.load(Ordering::Acquire);
    let y = PILL_Y.load(Ordering::Acquire);
    (x != i32::MIN && y != i32::MIN).then_some((x, y))
}

/// Clamp a position so the whole pill stays on the cached monitor, honouring
/// the monitor's origin (nonzero on multi-monitor layouts). An unknown monitor
/// size (0) leaves that axis unbounded.
fn clamp_pill_pos(x: i32, y: i32) -> (i32, i32) {
    let mon_x = MON_X.load(Ordering::Acquire);
    let mon_y = MON_Y.load(Ordering::Acquire);
    let mon_w = MON_W.load(Ordering::Acquire);
    let mon_h = MON_H.load(Ordering::Acquire);
    let (min_x, max_x) = if mon_w > 0 {
        (mon_x, mon_x + (mon_w - PILL_W as i32).max(0))
    } else {
        (i32::MIN, i32::MAX)
    };
    let (min_y, max_y) = if mon_h > 0 {
        (mon_y, mon_y + (mon_h - PILL_H as i32).max(0))
    } else {
        (i32::MIN, i32::MAX)
    };
    (x.clamp(min_x, max_x), y.clamp(min_y, max_y))
}

/// Logical bounds of the monitor whose area contains the pill's centre at
/// (x, y), or the nearest monitor if none does. `None` if the monitor list
/// can't be read. Main thread (enumerates monitors).
fn monitor_bounds_at(pill: &tauri::WebviewWindow, x: i32, y: i32) -> Option<(i32, i32, i32, i32)> {
    let cx = x + PILL_W as i32 / 2;
    let cy = y + PILL_H as i32 / 2;
    let mut contained: Option<(i32, i32, i32, i32)> = None;
    let mut nearest: Option<(i64, (i32, i32, i32, i32))> = None;
    for m in pill.available_monitors().ok()? {
        let p = m.position();
        let s = m.size();
        let scale = m.scale_factor();
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        let b = (
            (p.x as f64 / scale) as i32,
            (p.y as f64 / scale) as i32,
            (s.width as f64 / scale) as i32,
            (s.height as f64 / scale) as i32,
        );
        let (mx, my, mw, mh) = b;
        if cx >= mx && cx < mx + mw && cy >= my && cy < my + mh {
            contained = Some(b);
            break;
        }
        let ccx = mx + mw / 2;
        let ccy = my + mh / 2;
        let d = ((cx - ccx) as i64).pow(2) + ((cy - ccy) as i64).pow(2);
        if nearest.is_none_or(|(bd, _)| d < bd) {
            nearest = Some((d, b));
        }
    }
    contained.or_else(|| nearest.map(|(_, b)| b))
}

/// Clamp so the pill stays fully within the monitor its centre is over (or the
/// nearest one) — letting it cross freely between monitors in a multi-head
/// layout without vanishing into the gap between them. No clamp if the monitor
/// layout is unreadable. Global-coordinate path only; main thread.
fn clamp_to_current_monitor(pill: &tauri::WebviewWindow, x: i32, y: i32) -> (i32, i32) {
    let Some((mx, my, mw, mh)) = monitor_bounds_at(pill, x, y) else {
        return (x, y);
    };
    let max_x = (mx + (mw - PILL_W as i32)).max(mx);
    let max_y = (my + (mh - PILL_H as i32)).max(my);
    (x.clamp(mx, max_x), y.clamp(my, max_y))
}

/// Clamp a desired pill position on-screen. On the global-coordinate path
/// (X11/XWayland, Windows, macOS) uses the live monitor layout so drags cross
/// monitors cleanly; on the layer-shell path uses the cached per-output bounds.
/// Main thread.
fn clamp_pill_target(pill: &tauri::WebviewWindow, x: i32, y: i32) -> (i32, i32) {
    if global_coords() {
        clamp_to_current_monitor(pill, x, y)
    } else {
        clamp_pill_pos(x, y)
    }
}

/// Whether pill positions are in the compositor's global coordinate space
/// (X11/XWayland, Windows, macOS — `set_position`), as opposed to per-output
/// layer-shell margins (COSMIC/wlroots). The Linux check touches GTK, so
/// main-thread only.
fn global_coords() -> bool {
    #[cfg(target_os = "linux")]
    {
        !use_layer_shell()
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// Logical bounds (origin + size) of the monitor the pill is on, in the global
/// multi-monitor coordinate space `set_position` uses. Physical values ÷ scale.
fn monitor_logical_bounds(pill: &tauri::WebviewWindow) -> Option<(i32, i32, i32, i32)> {
    let monitor = pill
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| pill.primary_monitor().ok().flatten())?;
    let pos = monitor.position();
    let size = monitor.size();
    let scale = monitor.scale_factor();
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    Some((
        (pos.x as f64 / scale) as i32,
        (pos.y as f64 / scale) as i32,
        (size.width as f64 / scale) as i32,
        (size.height as f64 / scale) as i32,
    ))
}

/// Resolve the pill's starting position: the saved/live value if set, else a
/// default bottom-centre placement computed from the monitor. Caches the
/// monitor size for later drag clamping and stores the resolved position.
fn resolve_pill_position(pill: &tauri::WebviewWindow) -> (i32, i32) {
    if let Some((mx, my, mw, mh)) = monitor_logical_bounds(pill) {
        // Layer-shell margins are per-output (origin 0); global coords keep the
        // monitor's offset so multi-monitor placement lands on the right screen.
        let (ox, oy) = if global_coords() { (mx, my) } else { (0, 0) };
        MON_X.store(ox, Ordering::Release);
        MON_Y.store(oy, Ordering::Release);
        MON_W.store(mw, Ordering::Release);
        MON_H.store(mh, Ordering::Release);
    }
    let (x, y) = match current_pill_position() {
        Some((x, y)) => clamp_pill_target(pill, x, y),
        None => {
            // Default placement: horizontally centred, near the bottom with a
            // gap so it clears a taskbar/dock — offset by the monitor origin so
            // it lands on the intended screen in a multi-monitor layout.
            let mon_x = MON_X.load(Ordering::Acquire);
            let mon_y = MON_Y.load(Ordering::Acquire);
            let mon_w = MON_W.load(Ordering::Acquire);
            let mon_h = MON_H.load(Ordering::Acquire);
            let x = mon_x + ((mon_w - PILL_W as i32) / 2).max(0);
            let y = (mon_y + mon_h - PILL_H as i32 - PILL_DEFAULT_BOTTOM_GAP).max(mon_y);
            (x, y)
        }
    };
    PILL_X.store(x, Ordering::Release);
    PILL_Y.store(y, Ordering::Release);
    (x, y)
}

/// Move the pill to a logical position: layer-shell margins on Wayland,
/// `set_position` on X11/Windows/macOS. Must run on the main thread.
fn apply_pill_position(pill: &tauri::WebviewWindow, x: i32, y: i32) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    if use_layer_shell() {
        use gtk::prelude::*;
        use gtk_layer_shell::LayerShell;
        let gtk_window: gtk::Window = pill.gtk_window().map_err(|e| e.to_string())?.upcast();
        gtk_window.set_layer_shell_margin(gtk_layer_shell::Edge::Left, x);
        gtk_window.set_layer_shell_margin(gtk_layer_shell::Edge::Top, y);
        return Ok(());
    }
    pill.set_position(tauri::LogicalPosition::new(x, y))
        .map_err(|e| e.to_string())
}

/// Single source of truth for whether the pill window should be on screen,
/// given the live service status and the visibility setting. Called only from
/// Rust (startup + publish_status) — the frontend never decides visibility, so
/// the mapped window and its content can't desync. `Never` also destroys the
/// window (see set_pill_enabled); it returns false here defensively.
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

/// Whether the *current* pill window has had its Wayland layer-shell surface
/// configured. An `AtomicBool` rather than a `OnceLock` precisely because the
/// window can be destroyed and rebuilt when the user toggles pill visibility —
/// each new window must be re-configured on its first show, so create/destroy
/// reset this to false.
#[cfg(target_os = "linux")]
static LAYER_SHELL_CONFIGURED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[tauri::command]
pub async fn show_pill_window(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    run_pill_window_on_main_thread(&app, move |pill| {
        #[cfg(target_os = "linux")]
        {
            if !LAYER_SHELL_CONFIGURED.load(std::sync::atomic::Ordering::Acquire) {
                configure_linux_pill(&pill)?;
                LAYER_SHELL_CONFIGURED.store(true, std::sync::atomic::Ordering::Release);
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // TODO(macos): Use NSPanel with floatingPanel + NSNonactivatingPanelMask
            // for proper overlay behavior. Standard always_on_top shows in Mission Control.
            reposition_standard(&pill)?;
        }

        pill.show().map_err(|e| e.to_string())?;
        Ok(())
    })
    .await
}

#[tauri::command]
pub async fn hide_pill_window(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    run_pill_window_on_main_thread(&app, move |pill| {
        pill.hide().map_err(|e| e.to_string())?;
        Ok(())
    })
    .await
}

#[tauri::command]
pub async fn reposition_pill_center(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    run_pill_window_on_main_thread(&app, move |pill| {
        #[cfg(target_os = "linux")]
        {
            // Layer-shell repositioning is handled by anchors — no-op. Otherwise
            // (X11, or Wayland without layer-shell) fall back to standard positioning.
            if !use_layer_shell() {
                reposition_standard(&pill)?;
            }
        }

        #[cfg(not(target_os = "linux"))]
        reposition_standard(&pill)?;

        Ok(())
    })
    .await
}

/// Move the pill by a drag delta (logical px), clamped on-screen. Called
/// repeatedly from the pill's pointer-drag handler; the final resting position
/// is persisted separately by `save_pill_position` on pointer-up.
#[tauri::command]
pub async fn nudge_pill(
    dx: i32,
    dy: i32,
    window: WebviewWindow,
    app: AppHandle,
) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    let desired_x = PILL_X.load(Ordering::Acquire).saturating_add(dx);
    let desired_y = PILL_Y.load(Ordering::Acquire).saturating_add(dy);
    run_pill_window_on_main_thread(&app, move |pill| {
        // Clamp against the live monitor layout (needs the main thread) so a
        // drag across monitors stays on-screen instead of vanishing into the
        // gap between them, then apply.
        let (x, y) = clamp_pill_target(&pill, desired_x, desired_y);
        PILL_X.store(x, Ordering::Release);
        PILL_Y.store(y, Ordering::Release);
        apply_pill_position(&pill, x, y)
    })
    .await
}

/// Anchor for a cursor-pinned drag on the X11/XWayland path: the pointer's
/// root position and the pill's logical position at drag start, plus the
/// window scale. Every `drag_pill_tick` re-derives the pill position
/// *absolutely* from the live pointer against this anchor, so per-event
/// rounding and dropped events can never accumulate into drift — the failure
/// mode of the incremental `nudge_pill` path, whose integer `movementX/Y`
/// deltas made the pill wander off the cursor over a long drag.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
struct DragAnchor {
    pointer: (i32, i32),
    pill: (i32, i32),
    scale: f64,
}

#[cfg(target_os = "linux")]
static DRAG_ANCHOR: std::sync::Mutex<Option<DragAnchor>> = std::sync::Mutex::new(None);

/// Pointer position in root (device) coordinates via the default GDK seat.
/// Meaningful only on X11/XWayland — Wayland exposes no global pointer, which
/// is why the layer-shell path keeps the incremental fallback. Main thread.
#[cfg(target_os = "linux")]
fn pointer_root_position() -> Option<(i32, i32)> {
    use gtk::gdk::prelude::*;
    let seat = gtk::gdk::Display::default()?.default_seat()?;
    let (_screen, x, y) = seat.pointer()?.position();
    Some((x, y))
}

/// Begin a cursor-pinned drag. `dx`/`dy` carry the displacement accumulated
/// before the drag threshold was crossed, folded into the anchor so the pill
/// first catches up to the original grab point. Returns whether native
/// pointer tracking is active; `false` (layer-shell, no pointer device, or an
/// uninitialised position) tells the frontend to keep the incremental
/// `nudge_pill` fallback.
#[tauri::command]
pub async fn begin_pill_drag(
    dx: i32,
    dy: i32,
    window: WebviewWindow,
    app: AppHandle,
) -> Result<bool, String> {
    require_window_access(&window, &["pill"])?;
    #[cfg(target_os = "linux")]
    {
        let handle = app.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        app.run_on_main_thread(move || {
            let anchor = (|| {
                if use_layer_shell() {
                    return None;
                }
                let pointer = pointer_root_position()?;
                let (px, py) = current_pill_position()?;
                let scale = handle
                    .get_webview_window("pill")?
                    .scale_factor()
                    .ok()
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .unwrap_or(1.0);
                Some(DragAnchor {
                    pointer,
                    pill: (px.saturating_add(dx), py.saturating_add(dy)),
                    scale,
                })
            })();
            let native = anchor.is_some();
            *DRAG_ANCHOR.lock().unwrap() = anchor;
            let _ = tx.send(native);
        })
        .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "begin pill drag did not complete".to_string())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (dx, dy, app);
        Ok(false)
    }
}

/// One tracking step of a cursor-pinned drag: place the pill at its anchor
/// offset from the *current* pointer position. Absolute, not incremental — a
/// missed or delayed tick leaves no residual error, the next tick lands
/// exactly. No-op when no drag is active.
#[tauri::command]
pub async fn drag_pill_tick(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    run_pill_window_on_main_thread(&app, move |pill| apply_drag_tick(&pill)).await
}

/// Finish a cursor-pinned drag: one final settle tick so the release position
/// is exact, then drop the anchor. The frontend persists the position via
/// `save_pill_position` afterwards.
#[tauri::command]
pub async fn end_pill_drag(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    run_pill_window_on_main_thread(&app, move |pill| {
        let result = apply_drag_tick(&pill);
        #[cfg(target_os = "linux")]
        {
            *DRAG_ANCHOR.lock().unwrap() = None;
        }
        result
    })
    .await
}

#[cfg(target_os = "linux")]
fn apply_drag_tick(pill: &tauri::WebviewWindow) -> Result<(), String> {
    let Some(anchor) = *DRAG_ANCHOR.lock().unwrap() else {
        return Ok(());
    };
    let Some((px, py)) = pointer_root_position() else {
        return Ok(());
    };
    let dx = (f64::from(px - anchor.pointer.0) / anchor.scale).round() as i32;
    let dy = (f64::from(py - anchor.pointer.1) / anchor.scale).round() as i32;
    let (x, y) = clamp_pill_target(pill, anchor.pill.0 + dx, anchor.pill.1 + dy);
    PILL_X.store(x, Ordering::Release);
    PILL_Y.store(y, Ordering::Release);
    apply_pill_position(pill, x, y)
}

#[cfg(not(target_os = "linux"))]
fn apply_drag_tick(_pill: &tauri::WebviewWindow) -> Result<(), String> {
    Ok(())
}

/// Persist the pill's current position to settings so it reopens there. Called
/// on pointer-up after a drag.
#[tauri::command]
pub async fn save_pill_position(window: WebviewWindow) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    let Some((x, y)) = current_pill_position() else {
        return Ok(());
    };
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let store = lune_persist::SettingsStore::new(paths.settings_file());
        let mut settings = store.load().map_err(|e| format!("load settings: {e}"))?;
        settings.pill_position = Some(lune_core::config::PillPosition { x, y });
        store
            .save(&settings)
            .map_err(|e| format!("save settings: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("save pill position task: {e}"))?
}

async fn run_pill_window_on_main_thread<F>(app: &AppHandle, task: F) -> Result<(), String>
where
    F: FnOnce(tauri::WebviewWindow) -> Result<(), String> + Send + 'static,
{
    let app = app.clone();
    let app_for_task = app.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();

    app.run_on_main_thread(move || {
        let result = match app_for_task.get_webview_window("pill") {
            Some(pill) => task(pill),
            None => Ok(()), // Pill window not created — silently skip
        };

        let _ = tx.send(result);
    })
    .map_err(|e| e.to_string())?;

    rx.await
        .map_err(|_| "pill main-thread task did not complete".to_string())?
}

/// Show the pill window directly from Rust — bypasses the frontend IPC roundtrip
/// so the hotkey response appears instant.
pub fn show_pill_direct(app: &AppHandle) {
    let outer = app.clone();
    let inner = app.clone();
    let _ = outer.run_on_main_thread(move || {
        if let Some(pill) = inner.get_webview_window("pill") {
            #[cfg(target_os = "linux")]
            if !LAYER_SHELL_CONFIGURED.load(std::sync::atomic::Ordering::Acquire)
                && configure_linux_pill(&pill).is_ok()
            {
                LAYER_SHELL_CONFIGURED.store(true, std::sync::atomic::Ordering::Release);
            }
            let _ = pill.show();
        }
    });
}

pub fn hide_pill_direct(app: &AppHandle) {
    let outer = app.clone();
    let inner = app.clone();
    let _ = outer.run_on_main_thread(move || {
        if let Some(pill) = inner.get_webview_window("pill") {
            let _ = pill.hide();
        }
    });
}

/// Create the pill overlay window (hidden). Idempotent — a no-op if it already
/// exists. **Must run on the main thread** (window creation requires it); at
/// startup `setup` is already on it, otherwise go through `set_pill_enabled`.
///
/// The window is created hidden and never mapped before its first show so the
/// Wayland layer-shell surface can be configured first (`configure_linux_pill`);
/// creating it here resets the layer-shell flag so a rebuilt window re-configures.
pub fn create_pill_window(app: &AppHandle) {
    if app.get_webview_window("pill").is_some() {
        return;
    }
    match WebviewWindowBuilder::new(app, "pill", WebviewUrl::App("pill.html".into()))
        .title("")
        .inner_size(PILL_W, PILL_H)
        .resizable(false)
        .decorations(false)
        // Transparent so the bar's rounded corners can show (an opaque window
        // is a hard rectangle). The bar content is fully opaque; only the four
        // corner triangles are transparent — the one transparency arrangement
        // that renders cleanly here. A partially-transparent *fill* still ghosts
        // or blacks out on this Nvidia + WebKitGTK stack (tauri #14924), so the
        // bar must stay fully opaque; only the corners may be see-through.
        .transparent(true)
        .shadow(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .focused(false)
        .visible(false)
        .build()
    {
        Ok(_) => {
            #[cfg(target_os = "linux")]
            LAYER_SHELL_CONFIGURED.store(false, std::sync::atomic::Ordering::Release);
        }
        Err(e) => tracing::error!(error = %e, "failed to create pill overlay window"),
    }
}

/// Destroy the pill overlay window, freeing its webview process. Used when the
/// user sets pill visibility to Never — hiding alone keeps a whole
/// WebKitWebProcess (~200 MB) alive. A no-op if the window doesn't exist.
/// **Must run on the main thread**; go through `set_pill_enabled` off it.
pub fn destroy_pill_window(app: &AppHandle) {
    if let Some(pill) = app.get_webview_window("pill") {
        if let Err(e) = pill.close() {
            tracing::warn!(error = %e, "failed to close pill overlay window");
        }
        #[cfg(target_os = "linux")]
        LAYER_SHELL_CONFIGURED.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Reconcile the pill window's existence with whether the pill is enabled
/// (visibility != Never), from any thread. Creates the hidden window when
/// enabled, destroys it (freeing the webview) when disabled. This is what makes
/// "Never" actually turn the pill off, and a later re-enable bring it back
/// without an app restart.
pub fn set_pill_enabled(app: &AppHandle, enabled: bool) {
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        if enabled {
            create_pill_window(&handle);
        } else {
            destroy_pill_window(&handle);
        }
    });
}

/// Standard window positioning (works on X11, Windows, macOS): honour the
/// saved/live position, else default bottom-centre. Shares
/// `resolve_pill_position` with the Wayland path so both respect a dragged
/// position.
fn reposition_standard(pill: &tauri::WebviewWindow) -> Result<(), String> {
    let (x, y) = resolve_pill_position(pill);
    pill.set_position(tauri::LogicalPosition::new(x, y))
        .map_err(|e| e.to_string())
}

#[cfg(target_os = "linux")]
fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
}

/// Whether to drive the pill as a Wayland layer-shell overlay. True only when
/// running on Wayland AND the compositor actually implements the
/// `wlr-layer-shell` protocol. GNOME/Mutter is Wayland but ships no layer-shell,
/// so calling `init_layer_shell()` there spams "compositor does not support Layer
/// Shell" CRITICALs, falls back to a plain XDG toplevel, and then silently
/// ignores our anchors/margins (breaking placement + drag). In that case we take
/// the standard XDG positioning path instead. Must run on the main thread
/// (`is_supported` asserts it).
#[cfg(target_os = "linux")]
fn use_layer_shell() -> bool {
    is_wayland() && gtk_layer_shell::is_supported()
}

/// Configure the pill as a layer shell overlay on Wayland, or use standard
/// positioning on X11.
#[cfg(target_os = "linux")]
fn configure_linux_pill(pill: &tauri::WebviewWindow) -> Result<(), String> {
    use gtk::prelude::*;
    use gtk_layer_shell::LayerShell;

    if !use_layer_shell() {
        // X11 / XWayland path (native Wayland without layer-shell runs under
        // XWayland — see run()). Pin the window to the pill footprint so
        // WebKitGTK can't grow it to the web content's natural height — a tall
        // transparent area renders black on this Nvidia + WebKitGTK stack.
        // always_on_top / skip_taskbar from the builder are honoured under X11.
        let gtk_window: gtk::Window = pill.gtk_window().map_err(|e| e.to_string())?.upcast();
        gtk_window.set_size_request(PILL_W as i32, PILL_H as i32);
        // WebKitGTK also loses page alpha entirely under XWayland on this GPU
        // stack, so the "transparent" corner pixels around the rounded bar
        // paint opaque black. Clip the window itself to the bar's rounded rect
        // (XShape) so those pixels never reach the screen. Realize first — the
        // shape is applied to the widget's realized GDK window.
        gtk_window.realize();
        gtk_window.shape_combine_region(Some(&rounded_pill_region()));
        return reposition_standard(pill);
    }

    let app_window = pill.gtk_window().map_err(|e| e.to_string())?;
    let gtk_window: gtk::Window = app_window.upcast();

    gtk_window.init_layer_shell();
    gtk_window.set_layer(gtk_layer_shell::Layer::Overlay);
    gtk_window.set_namespace("lune-voice-pill");

    // Don't take keyboard focus — pill has click buttons only.
    gtk_window.set_keyboard_mode(gtk_layer_shell::KeyboardMode::None);

    // Anchor the top-left corner and drive position via margins. Layer-shell
    // surfaces can't be moved by the compositor, so absolute placement (and
    // dragging) is expressed as one anchored corner plus left/top margins.
    gtk_window.set_anchor(gtk_layer_shell::Edge::Top, true);
    gtk_window.set_anchor(gtk_layer_shell::Edge::Left, true);
    gtk_window.set_anchor(gtk_layer_shell::Edge::Right, false);
    gtk_window.set_anchor(gtk_layer_shell::Edge::Bottom, false);

    // -1 = don't reserve exclusive space (don't push other windows).
    gtk_window.set_exclusive_zone(-1);

    // Pin the surface to the pill footprint. The layer surface height is driven
    // by the embedded WebKitWebView's natural size and can't be forced smaller
    // here (resize is ignored), so the short-bar height is pinned by
    // `.pill-root`'s fixed CSS height (which PILL_H mirrors); width is honoured
    // from inner_size.
    gtk_window.set_size_request(PILL_W as i32, PILL_H as i32);

    // Place it: saved/last position, else default bottom-centre.
    let (x, y) = resolve_pill_position(pill);
    gtk_window.set_layer_shell_margin(gtk_layer_shell::Edge::Left, x);
    gtk_window.set_layer_shell_margin(gtk_layer_shell::Edge::Top, y);

    Ok(())
}

/// The pill bar's rounded-rect footprint as a cairo region: one 1px-tall span
/// per row, with corner rows inset by the circle equation. `ceil` biases the
/// clip inward so no black corner pixel survives at the antialiased edge.
#[cfg(target_os = "linux")]
fn rounded_pill_region() -> gtk::cairo::Region {
    use gtk::cairo::{RectangleInt, Region};
    let (w, h, r) = (PILL_W as i32, PILL_H as i32, PILL_RADIUS);
    let region = Region::create();
    for y in 0..h {
        let cy = f64::from(y) + 0.5;
        let d = if cy < f64::from(r) {
            f64::from(r) - cy
        } else if cy > f64::from(h - r) {
            cy - f64::from(h - r)
        } else {
            0.0
        };
        let inset = if d > 0.0 {
            (f64::from(r) - (f64::from(r * r) - d * d).max(0.0).sqrt()).ceil() as i32
        } else {
            0
        };
        let _ = region.union_rectangle(&RectangleInt::new(inset, y, w - 2 * inset, 1));
    }
    region
}

#[cfg(test)]
mod tests {
    use super::should_show_pill;
    use lune_core::config::PillVisibility::{Always, Never, WhenDictating};

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
