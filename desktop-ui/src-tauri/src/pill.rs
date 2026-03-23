use tauri::{AppHandle, Manager, WebviewWindow};

use crate::commands::require_window_access;

#[cfg(target_os = "linux")]
static LAYER_SHELL_CONFIGURED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

#[tauri::command]
pub async fn show_pill_window(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    run_pill_window_on_main_thread(&app, move |pill| {
        #[cfg(target_os = "linux")]
        {
            if LAYER_SHELL_CONFIGURED.get().is_none() {
                configure_linux_pill(&pill)?;
                let _ = LAYER_SHELL_CONFIGURED.set(());
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

async fn run_pill_window_on_main_thread<F>(app: &AppHandle, task: F) -> Result<(), String>
where
    F: FnOnce(tauri::WebviewWindow) -> Result<(), String> + Send + 'static,
{
    let app = app.clone();
    let app_for_task = app.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();

    app.run_on_main_thread(move || {
        let result = (|| {
            let pill = app_for_task
                .get_webview_window("pill")
                .ok_or_else(|| "pill window not found".to_string())?;
            task(pill)
        })();

        let _ = tx.send(result);
    })
    .map_err(|e| e.to_string())?;

    rx.await
        .map_err(|_| "pill main-thread task did not complete".to_string())?
}

/// Standard window positioning (works on X11, Windows, macOS).
fn reposition_standard(pill: &tauri::WebviewWindow) -> Result<(), String> {
    let monitor = pill
        .current_monitor()
        .map_err(|e| e.to_string())?
        .or_else(|| pill.primary_monitor().ok().flatten())
        .ok_or_else(|| "no monitor found".to_string())?;

    let screen = monitor.size();
    let scale = monitor.scale_factor();
    let pill_size = pill.outer_size().map_err(|e| e.to_string())?;

    let screen_w = screen.width as f64 / scale;
    let screen_h = screen.height as f64 / scale;
    let pill_w = pill_size.width as f64 / scale;
    let pill_h = pill_size.height as f64 / scale;

    let x = ((screen_w - pill_w) / 2.0) as i32;
    let y = (screen_h - pill_h - 40.0) as i32;

    pill.set_position(tauri::LogicalPosition::new(x, y))
        .map_err(|e| e.to_string())?;

    Ok(())
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

    // Anchor bottom only → compositor centers horizontally.
    gtk_window.set_anchor(gtk_layer_shell::Edge::Bottom, true);
    gtk_window.set_anchor(gtk_layer_shell::Edge::Left, false);
    gtk_window.set_anchor(gtk_layer_shell::Edge::Right, false);
    gtk_window.set_anchor(gtk_layer_shell::Edge::Top, false);

    gtk_window.set_layer_shell_margin(gtk_layer_shell::Edge::Bottom, 40);

    // -1 = don't reserve exclusive space (don't push other windows).
    gtk_window.set_exclusive_zone(-1);

    Ok(())
}
