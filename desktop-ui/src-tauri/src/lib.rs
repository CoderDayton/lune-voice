mod commands;
mod dto;
mod persistence;
mod pill;
mod runtime;
#[cfg(test)]
mod tauri_integration_tests;

use tauri::Manager;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let runtime =
                tauri::async_runtime::block_on(runtime::AppRuntime::new(app.handle().clone()))
                    .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            app.manage(runtime);

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
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.unminimize();
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
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
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.unminimize();
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
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
            commands::delete_dictionary_entry,
            commands::subscribe_session_stream,
            commands::detach_session_stream,
            commands::start_dictation,
            commands::stop_dictation,
            commands::cancel_dictation,
            commands::toggle_dictation,
            pill::show_pill_window,
            pill::hide_pill_window,
            pill::reposition_pill_center
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            eprintln!("fatal: tauri application exited with error: {e}");
            std::process::exit(1);
        });
}
