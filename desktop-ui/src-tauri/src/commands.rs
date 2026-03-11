use crate::dto::{CreateDictionaryEntryInput, WorkspaceSettingsDto, WorkspaceSnapshot};

#[tauri::command]
pub fn ping() -> &'static str {
    "pong"
}

#[tauri::command]
pub async fn load_workspace_snapshot() -> Result<WorkspaceSnapshot, String> {
    tauri::async_runtime::spawn_blocking(crate::persistence::load_workspace_snapshot)
        .await
        .map_err(|e| format!("load workspace snapshot task failed: {e}"))?
}

#[tauri::command]
pub async fn save_workspace_settings(
    input: WorkspaceSettingsDto,
) -> Result<WorkspaceSnapshot, String> {
    tauri::async_runtime::spawn_blocking(move || crate::persistence::save_workspace_settings(input))
        .await
        .map_err(|e| format!("save workspace settings task failed: {e}"))?
}

#[tauri::command]
pub async fn create_dictionary_entry(
    input: CreateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    tauri::async_runtime::spawn_blocking(move || crate::persistence::create_dictionary_entry(input))
        .await
        .map_err(|e| format!("create dictionary entry task failed: {e}"))?
}

#[tauri::command]
pub async fn delete_dictionary_entry(entry_id: i32) -> Result<WorkspaceSnapshot, String> {
    tauri::async_runtime::spawn_blocking(move || {
        crate::persistence::delete_dictionary_entry(entry_id)
    })
    .await
    .map_err(|e| format!("delete dictionary entry task failed: {e}"))?
}
