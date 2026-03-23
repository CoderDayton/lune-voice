mod commands;
mod dto;
mod persistence;
mod pill;
mod runtime;
#[cfg(test)]
mod tauri_integration_tests;

use tauri::Manager;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let runtime =
                tauri::async_runtime::block_on(runtime::AppRuntime::new(app.handle().clone()))
                    .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            app.manage(runtime);
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
