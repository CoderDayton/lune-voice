mod commands;
mod dto;
#[cfg(target_os = "linux")]
mod hotkey_linux;
mod persistence;
mod pill;
mod runtime;
#[cfg(test)]
mod tauri_integration_tests;

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
#[cfg(not(target_os = "linux"))]
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

/// Set when the user explicitly requested a quit (tray "Quit" menu).
/// Without this flag, the ExitRequested handler can't tell intentional
/// quits apart from window-closes, and we'd block both.
static QUIT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Frontend-invokable quit. Mirrors the tray "Quit" path: set the intent
/// flag so the ExitRequested handler honors the exit, then terminate.
#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    QUIT_REQUESTED.store(true, Ordering::SeqCst);
    app.exit(0);
}

/// Smallest usable window. Also the OS-enforced minimum and the floor for the
/// adaptive default, so the UI never opens below its designed layout width.
const MIN_WINDOW_W: f64 = 1280.0;
const MIN_WINDOW_H: f64 = 720.0;
/// Upper bound on the adaptive default — caps ultrawide and 4K monitors so the
/// window opens at a comfortable size instead of a sprawling one.
const MAX_WINDOW_W: f64 = 1600.0;
const MAX_WINDOW_H: f64 = 1100.0;
/// Fraction of the monitor the default window aims to fill (~70%, mid the
/// desired 60–75% range).
const WINDOW_SCREEN_FRACTION: f64 = 0.70;

/// Adaptive default window size: ~70% of the primary monitor's logical size,
/// clamped to `[MIN, MAX]` per axis. The width cap keeps ultrawides from
/// opening an absurdly wide window; the floor keeps small laptops at the UI's
/// designed minimum. Falls back to the min size if the monitor can't be read.
fn adaptive_window_size(app: &tauri::AppHandle) -> (f64, f64) {
    let Ok(Some(monitor)) = app.primary_monitor() else {
        return (MIN_WINDOW_W, MIN_WINDOW_H);
    };
    // Guard against a bogus scale (0 / NaN / inf) so the division can't yield a
    // non-finite size the builder would choke on.
    let raw_scale = monitor.scale_factor();
    let scale = if raw_scale.is_finite() && raw_scale > 0.0 {
        raw_scale
    } else {
        1.0
    };
    let size = monitor.size();
    let logical_w = size.width as f64 / scale;
    let logical_h = size.height as f64 / scale;
    let width = (logical_w * WINDOW_SCREEN_FRACTION).clamp(MIN_WINDOW_W, MAX_WINDOW_W);
    let height = (logical_h * WINDOW_SCREEN_FRACTION).clamp(MIN_WINDOW_H, MAX_WINDOW_H);
    (width, height)
}

/// Bring the main window to front, creating it on demand if it doesn't exist.
/// We don't keep the window alive while minimized — closing destroys the
/// webview to free its memory, and the next call here rebuilds it.
fn create_or_focus_main_window(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
        return;
    }
    let (width, height) = adaptive_window_size(app);
    if let Err(e) = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .title("Lune Voice")
        .inner_size(width, height)
        .min_inner_size(MIN_WINDOW_W, MIN_WINDOW_H)
        .resizable(true)
        .decorations(true)
        .center()
        .build()
    {
        tracing::error!(error = %e, "failed to create main window");
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            create_or_focus_main_window(app);
        }))
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let runtime =
                tauri::async_runtime::block_on(runtime::AppRuntime::new(app.handle().clone()))
                    .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            let start_minimized = tauri::async_runtime::block_on(runtime.start_minimized());
            app.manage(runtime);

            // Build the main window only when not starting minimized. While
            // minimized the webview isn't created at all — the tray icon
            // (registered below) is the way to bring it up, which lazily
            // builds it via create_or_focus_main_window.
            if !start_minimized {
                create_or_focus_main_window(app.handle());
            }

            // ── Global hotkeys ─────────────────────────────────────────────
            // Linux: evdev grab works on both X11 and Wayland.
            // Other platforms: tauri-plugin-global-shortcut.
            #[cfg(target_os = "linux")]
            spawn_evdev_hotkeys(app)?;
            #[cfg(not(target_os = "linux"))]
            register_hotkeys(app)?;

            // ── System tray ──────────────────────────────────────────────
            let show_i = MenuItem::with_id(app, "show", "Open Lune Voice", true, None::<&str>)?;
            let toggle_i = MenuItem::with_id(app, "toggle", "Start Dictation", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &toggle_i, &sep, &quit_i])?;

            TrayIconBuilder::new()
                .icon(
                    app.default_window_icon()
                        .cloned()
                        .ok_or("no default window icon — check tauri.conf.json icons")?,
                )
                .tooltip("Lune Voice")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => {
                        create_or_focus_main_window(app);
                    }
                    "toggle" => {
                        let rt = app.state::<runtime::AppRuntime>();
                        let rt = rt.inner().clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = rt.toggle_dictation().await;
                        });
                    }
                    "quit" => {
                        QUIT_REQUESTED.store(true, Ordering::SeqCst);
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        create_or_focus_main_window(app);
                    }
                })
                .build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::load_workspace_snapshot,
            commands::load_pill_snapshot,
            commands::save_workspace_settings,
            commands::create_dictionary_entry,
            commands::delete_session,
            commands::list_history_page,
            commands::delete_dictionary_entry,
            commands::update_dictionary_entry,
            commands::create_custom_style,
            commands::update_custom_style,
            commands::delete_custom_style,
            commands::generate_style_examples,
            commands::subscribe_session_stream,
            commands::detach_session_stream,
            commands::start_dictation,
            commands::stop_dictation,
            commands::cancel_dictation,
            commands::toggle_dictation,
            commands::calibrate_start,
            commands::calibrate_advance,
            commands::calibrate_cancel,
            pill::show_pill_window,
            pill::hide_pill_window,
            pill::reposition_pill_center,
            commands::list_audio_input_devices,
            commands::check_model_status,
            commands::download_model_cmd,
            commands::delete_model_cmd,
            commands::initialize_pipeline,
            commands::gpu_support,
            quit_app
        ])
        .build(tauri::generate_context!())
        .unwrap_or_else(|e| {
            eprintln!("fatal: tauri application exited with error: {e}");
            std::process::exit(1);
        })
        .run(|_app, event| {
            // Closing the main window destroys it to free webview memory.
            // Without this, Tauri would also exit the app. We keep the
            // process alive so the tray icon, hotkeys, and dictation
            // runtime continue working — the user can rebuild the window
            // from the tray. Tray "Quit" sets QUIT_REQUESTED so its
            // explicit app.exit(0) is honored.
            if let tauri::RunEvent::ExitRequested { api, .. } = event
                && !QUIT_REQUESTED.load(Ordering::SeqCst)
            {
                api.prevent_exit();
            }
        });
}

#[cfg(not(target_os = "linux"))]
fn register_hotkeys(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // Read hotkey config directly — can't use block_on here since we're
    // already inside one from setup(). Just use the defaults; the runtime
    // has the same settings loaded but we can't await inside setup.
    let settings = lune_core::config::UserSettings::default();
    if let Some(paths) = lune_persist::AppPaths::resolve() {
        let store = lune_persist::SettingsStore::new(paths.settings_file());
        if let Ok(s) = store.load() {
            return register_hotkeys_with(app, &s);
        }
    }
    register_hotkeys_with(app, &settings)
}

#[cfg(not(target_os = "linux"))]
fn register_hotkeys_with(
    app: &tauri::App,
    settings: &lune_core::config::UserSettings,
) -> Result<(), Box<dyn std::error::Error>> {
    let toggle_str = format_hotkey(&settings.hotkey.toggle_dictation);
    let cancel_str = format_hotkey(&settings.hotkey.cancel_dictation);

    let toggle: Shortcut = toggle_str
        .parse()
        .map_err(|e| format!("parse toggle hotkey: {e}"))?;
    let cancel: Shortcut = cancel_str
        .parse()
        .map_err(|e| format!("parse cancel hotkey: {e}"))?;

    app.global_shortcut()
        .on_shortcut(toggle, move |app, _shortcut, event| {
            if event.state == ShortcutState::Pressed {
                let rt = app.state::<runtime::AppRuntime>();
                let rt = rt.inner().clone();
                tauri::async_runtime::spawn(async move {
                    let _ = rt.toggle_dictation().await;
                });
            }
        })?;

    app.global_shortcut()
        .on_shortcut(cancel, move |app, _shortcut, event| {
            if event.state == ShortcutState::Pressed {
                let rt = app.state::<runtime::AppRuntime>();
                let rt = rt.inner().clone();
                tauri::async_runtime::spawn(async move {
                    let _ = rt.cancel_dictation().await;
                });
            }
        })?;

    tracing::info!(toggle = %toggle_str, cancel = %cancel_str, "global hotkeys registered");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn format_hotkey(combo: &lune_core::config::KeyCombo) -> String {
    let mut parts: Vec<&str> = combo
        .modifiers
        .iter()
        .map(|m| match m {
            lune_core::config::Modifier::Ctrl => "ctrl",
            lune_core::config::Modifier::Shift => "shift",
            lune_core::config::Modifier::Alt => "alt",
            lune_core::config::Modifier::Meta => "super",
            _ => "unknown",
        })
        .collect();
    parts.push(&combo.key);
    parts.join("+")
}

#[cfg(target_os = "linux")]
fn spawn_evdev_hotkeys(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let settings = {
        let mut s = lune_core::config::UserSettings::default();
        if let Some(paths) = lune_persist::AppPaths::resolve() {
            let store = lune_persist::SettingsStore::new(paths.settings_file());
            if let Ok(loaded) = store.load() {
                s = loaded;
            }
        }
        s
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);

    tracing::info!(
        toggle = ?settings.hotkey.toggle_dictation,
        cancel = ?settings.hotkey.cancel_dictation,
        mode = ?settings.activation_mode,
        "spawning evdev hotkey listener"
    );

    // Live config — managed in Tauri state so settings saves can hot-reload
    // every hotkey setting (trigger, modifiers, cancel, activation mode)
    // without restarting the listener thread.
    let Some(live) =
        hotkey_linux::LiveHotkeyConfig::new(&settings.hotkey, settings.activation_mode)
    else {
        tracing::warn!("evdev hotkey listener not started (unknown key name in config)");
        return Ok(());
    };
    let live = std::sync::Arc::new(live);
    app.manage(std::sync::Arc::clone(&live));

    match hotkey_linux::spawn(live, tx) {
        Some(_handle) => {
            // Thread handle intentionally leaked — runs for app lifetime.
            tracing::info!("evdev hotkey listener thread started");
        }
        None => {
            tracing::warn!("evdev hotkey listener thread failed to spawn");
            return Ok(());
        }
    }

    let rt = app.state::<runtime::AppRuntime>().inner().clone();
    tauri::async_runtime::spawn(async move {
        while let Some(action) = rx.recv().await {
            match action {
                hotkey_linux::HotkeyAction::Toggle => {
                    tracing::info!(action = "toggle", "hotkey pressed");
                    let _ = rt.toggle_dictation().await;
                }
                hotkey_linux::HotkeyAction::Start => {
                    tracing::info!(action = "start", "hotkey pressed");
                    let _ = rt.start_dictation().await;
                }
                hotkey_linux::HotkeyAction::Stop => {
                    tracing::info!(action = "stop", "hotkey pressed");
                    let _ = rt.stop_and_wait_dictation().await;
                }
                hotkey_linux::HotkeyAction::Cancel => {
                    if rt.is_active().await {
                        tracing::info!(action = "cancel", "hotkey pressed");
                        let _ = rt.cancel_dictation().await;
                    }
                }
            }
        }
        tracing::error!("hotkey channel closed — evdev listener has exited, hotkeys are dead");
    });

    tracing::info!("evdev hotkey listener spawned");
    Ok(())
}
