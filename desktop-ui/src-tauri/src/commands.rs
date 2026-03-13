use tauri::{State, ipc::Channel};

use crate::dto::{
    CreateDictionaryEntryInput, SessionStreamEvent, WorkspaceSettingsDto, WorkspaceSnapshot,
};
use crate::runtime::AppRuntime;

#[tauri::command]
pub fn ping() -> &'static str {
    "pong"
}

#[tauri::command]
pub async fn load_workspace_snapshot(
    runtime: State<'_, AppRuntime>,
) -> Result<WorkspaceSnapshot, String> {
    let mut snapshot =
        tauri::async_runtime::spawn_blocking(crate::persistence::load_workspace_snapshot)
            .await
            .map_err(|e| format!("load workspace snapshot task failed: {e}"))??;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn save_workspace_settings(
    runtime: State<'_, AppRuntime>,
    input: WorkspaceSettingsDto,
) -> Result<WorkspaceSnapshot, String> {
    let mut snapshot = tauri::async_runtime::spawn_blocking(move || {
        crate::persistence::save_workspace_settings(input)
    })
    .await
    .map_err(|e| format!("save workspace settings task failed: {e}"))??;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn create_dictionary_entry(
    runtime: State<'_, AppRuntime>,
    input: CreateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    let mut snapshot = tauri::async_runtime::spawn_blocking(move || {
        crate::persistence::create_dictionary_entry(input)
    })
    .await
    .map_err(|e| format!("create dictionary entry task failed: {e}"))??;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn delete_dictionary_entry(
    runtime: State<'_, AppRuntime>,
    entry_id: i32,
) -> Result<WorkspaceSnapshot, String> {
    let mut snapshot = tauri::async_runtime::spawn_blocking(move || {
        crate::persistence::delete_dictionary_entry(entry_id)
    })
    .await
    .map_err(|e| format!("delete dictionary entry task failed: {e}"))??;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn subscribe_session_stream(
    runtime: State<'_, AppRuntime>,
    channel: Channel<SessionStreamEvent>,
) -> Result<(), String> {
    runtime.set_session_stream(channel).await
}

#[tauri::command]
pub async fn detach_session_stream(runtime: State<'_, AppRuntime>) -> Result<(), String> {
    runtime.clear_session_stream().await;
    Ok(())
}

#[tauri::command]
pub async fn start_dictation(runtime: State<'_, AppRuntime>) -> Result<String, String> {
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.start_dictation())
    })
    .await
    .map_err(|e| format!("start dictation task panicked: {e}"))?
}

#[tauri::command]
pub async fn stop_dictation(runtime: State<'_, AppRuntime>) -> Result<(), String> {
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.stop_dictation())
    })
    .await
    .map_err(|e| format!("stop dictation task panicked: {e}"))?
}

#[tauri::command]
pub async fn cancel_dictation(runtime: State<'_, AppRuntime>) -> Result<(), String> {
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.cancel_dictation())
    })
    .await
    .map_err(|e| format!("cancel dictation task panicked: {e}"))?
}

#[tauri::command]
pub async fn toggle_dictation(runtime: State<'_, AppRuntime>) -> Result<Option<String>, String> {
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.toggle_dictation())
    })
    .await
    .map_err(|e| format!("toggle dictation task panicked: {e}"))?
}
