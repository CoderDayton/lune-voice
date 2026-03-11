mod commands;
mod dto;
mod persistence;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::ping,
            commands::load_workspace_snapshot,
            commands::save_workspace_settings,
            commands::create_dictionary_entry,
            commands::delete_dictionary_entry
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
