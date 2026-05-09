mod commands;
mod dto;
#[cfg(target_os = "linux")]
mod hotkey_linux;
mod persistence;
mod pill;
mod runtime;
#[cfg(test)]
mod tauri_integration_tests;

use tauri::Manager;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
#[cfg(not(target_os = "linux"))]
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

/// Show, unminimize, and focus the main application window.
fn show_main_window(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app);
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

            // Hide the main window when start_minimized is on. The tray icon
            // (registered below) remains the way to bring it back up.
            if start_minimized && let Some(w) = app.get_webview_window("main") {
                let _ = w.hide();
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
                        show_main_window(app);
                    }
                    "toggle" => {
                        let rt = app.state::<runtime::AppRuntime>();
                        let rt = rt.inner().clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = rt.toggle_dictation().await;
                        });
                    }
                    "quit" => {
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
                        show_main_window(app);
                    }
                })
                .build(app)?;

            // ── Close to tray (hide instead of quit) ─────────────────────
            if let Some(main_window) = app.get_webview_window("main") {
                let w = main_window.clone();
                main_window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = w.hide();
                    }
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::load_workspace_snapshot,
            commands::load_pill_snapshot,
            commands::save_workspace_settings,
            commands::create_dictionary_entry,
            commands::delete_session,
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
            commands::check_model_status,
            commands::download_model_cmd,
            commands::delete_model_cmd,
            commands::initialize_pipeline
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            eprintln!("fatal: tauri application exited with error: {e}");
            std::process::exit(1);
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

    match hotkey_linux::spawn(settings.hotkey, settings.activation_mode, tx) {
        Some(_handle) => {
            // Thread handle intentionally leaked — runs for app lifetime.
            tracing::info!("evdev hotkey listener thread started");
        }
        None => {
            tracing::warn!("evdev hotkey listener failed to start (unknown key name in config?)");
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
