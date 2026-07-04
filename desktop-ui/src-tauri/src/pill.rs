use lune_core::config::PillVisibility;
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

use crate::commands::require_window_access;
use std::sync::atomic::{AtomicI32, Ordering};

/// Fixed pill overlay footprint — a wide, short rounded bar. Fixed for all modes
/// to avoid compositor ghost artifacts on Wayland windows during resize (wry
/// #1667, tauri #14924). The window IS the pill's footprint; the webview content
/// fills it edge-to-edge (`.pill-root` is a fixed 32px tall), and the layer-shell
/// surface height follows that intrinsic content height, so PILL_H must match it.
const PILL_W: f64 = 104.0;
const PILL_H: f64 = 32.0;

/// Live pill position (logical px from the monitor's top-left). `i32::MIN` =
/// uninitialised — `resolve_pill_position` then loads the saved value or
/// computes the default. Applied on Wayland via layer-shell margins, elsewhere
/// via `set_position`.
static PILL_X: AtomicI32 = AtomicI32::new(i32::MIN);
static PILL_Y: AtomicI32 = AtomicI32::new(i32::MIN);
/// Cached logical monitor size, for clamping drags on-screen (0 = unknown).
static MON_W: AtomicI32 = AtomicI32::new(0);
static MON_H: AtomicI32 = AtomicI32::new(0);

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

/// Clamp a position so the whole pill stays on the cached monitor. An unknown
/// monitor size (0) leaves that axis unbounded.
fn clamp_pill_pos(x: i32, y: i32) -> (i32, i32) {
    let mon_w = MON_W.load(Ordering::Acquire);
    let mon_h = MON_H.load(Ordering::Acquire);
    let max_x = if mon_w > 0 {
        (mon_w - PILL_W as i32).max(0)
    } else {
        i32::MAX
    };
    let max_y = if mon_h > 0 {
        (mon_h - PILL_H as i32).max(0)
    } else {
        i32::MAX
    };
    (x.clamp(0, max_x), y.clamp(0, max_y))
}

/// Logical size of the monitor the pill is on (physical size ÷ scale factor).
fn monitor_logical_size(pill: &tauri::WebviewWindow) -> Option<(i32, i32)> {
    let monitor = pill
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| pill.primary_monitor().ok().flatten())?;
    let size = monitor.size();
    let scale = monitor.scale_factor();
    Some((
        (size.width as f64 / scale) as i32,
        (size.height as f64 / scale) as i32,
    ))
}

/// Resolve the pill's starting position: the saved/live value if set, else a
/// default bottom-centre placement computed from the monitor. Caches the
/// monitor size for later drag clamping and stores the resolved position.
fn resolve_pill_position(pill: &tauri::WebviewWindow) -> (i32, i32) {
    if let Some((w, h)) = monitor_logical_size(pill) {
        MON_W.store(w, Ordering::Release);
        MON_H.store(h, Ordering::Release);
    }
    let (x, y) = match current_pill_position() {
        Some((x, y)) => clamp_pill_pos(x, y),
        None => {
            // Default placement: horizontally centred, and vertically a little
            // above the screen centre.
            let mon_w = MON_W.load(Ordering::Acquire);
            let mon_h = MON_H.load(Ordering::Acquire);
            let x = ((mon_w - PILL_W as i32) / 2).max(0);
            let y = (((mon_h - PILL_H as i32) / 2) - mon_h / 10).max(0);
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
    if is_wayland() {
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
            // On Wayland layer shell, repositioning is handled by anchors — no-op.
            // On X11, fall back to standard positioning.
            if !is_wayland() {
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
    let (x, y) = clamp_pill_pos(
        PILL_X.load(Ordering::Acquire).saturating_add(dx),
        PILL_Y.load(Ordering::Acquire).saturating_add(dy),
    );
    PILL_X.store(x, Ordering::Release);
    PILL_Y.store(y, Ordering::Release);
    run_pill_window_on_main_thread(&app, move |pill| apply_pill_position(&pill, x, y)).await
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

/// Configure the pill as a layer shell overlay on Wayland, or use standard
/// positioning on X11.
#[cfg(target_os = "linux")]
fn configure_linux_pill(pill: &tauri::WebviewWindow) -> Result<(), String> {
    use gtk::prelude::*;
    use gtk_layer_shell::LayerShell;

    if !is_wayland() {
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
