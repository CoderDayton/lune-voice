use tauri::{Emitter, Runtime, State, WebviewWindow, ipc::Channel};

use crate::dto::{
    CreateDictionaryEntryInput, PillSnapshotDto, SessionStreamEvent, WorkspaceSettingsDto,
    WorkspaceSnapshot,
};
use crate::runtime::AppRuntime;

pub(crate) fn require_window_access<R: Runtime>(
    window: &WebviewWindow<R>,
    allowed: &[&str],
) -> Result<(), String> {
    require_window_access_for_label(window.label(), allowed)
}

fn require_window_access_for_label(label: &str, allowed: &[&str]) -> Result<(), String> {
    if allowed.contains(&label) {
        Ok(())
    } else {
        Err(format!("window \"{label}\" cannot invoke this command"))
    }
}

#[tauri::command]
pub async fn load_workspace_snapshot(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot =
        tauri::async_runtime::spawn_blocking(crate::persistence::load_workspace_snapshot)
            .await
            .map_err(|e| format!("load workspace snapshot task failed: {e}"))??;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn load_pill_snapshot(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<PillSnapshotDto, String> {
    require_window_access(&window, &["pill"])?;
    let mut snapshot =
        tauri::async_runtime::spawn_blocking(crate::persistence::load_workspace_snapshot)
            .await
            .map_err(|e| format!("load pill snapshot task failed: {e}"))??;
    snapshot.service_status = runtime.service_status().await;
    Ok(PillSnapshotDto {
        service_status: snapshot.service_status,
        activation_mode: snapshot.settings.activation_mode,
        pill_visibility: snapshot.settings.pill_visibility,
    })
}

#[tauri::command]
pub async fn save_workspace_settings(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    input: WorkspaceSettingsDto,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = tauri::async_runtime::spawn_blocking(move || {
        crate::persistence::save_workspace_settings(input)
    })
    .await
    .map_err(|e| format!("save workspace settings task failed: {e}"))??;
    runtime.mark_reload_required().await;
    // Notify pill of settings changes (activation mode, pill visibility)
    if let Err(e) = runtime.app_handle().emit_to(
        "pill",
        "lune://pill-settings",
        serde_json::json!({
            "activationMode": &snapshot.settings.activation_mode,
            "pillVisibility": &snapshot.settings.pill_visibility,
        }),
    ) {
        tracing::debug!(error = %e, "pill settings emit skipped (pill window may not exist)");
    }
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn create_dictionary_entry(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    input: CreateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
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
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    entry_id: i32,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
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
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    channel: Channel<SessionStreamEvent>,
) -> Result<u64, String> {
    require_window_access(&window, &["main"])?;
    runtime.set_session_stream(channel).await
}

#[tauri::command]
pub async fn detach_session_stream(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    subscription_id: u64,
) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    runtime.clear_session_stream(subscription_id).await;
    Ok(())
}

#[tauri::command]
pub async fn start_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<String, String> {
    require_window_access(&window, &["pill"])?;
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.start_dictation())
    })
    .await
    .map_err(|e| format!("start dictation task panicked: {e}"))?
}

#[tauri::command]
pub async fn stop_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.stop_dictation())
    })
    .await
    .map_err(|e| format!("stop dictation task panicked: {e}"))?
}

#[tauri::command]
pub async fn cancel_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<(), String> {
    require_window_access(&window, &["pill"])?;
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.cancel_dictation())
    })
    .await
    .map_err(|e| format!("cancel dictation task panicked: {e}"))?
}

#[tauri::command]
pub async fn toggle_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<Option<String>, String> {
    require_window_access(&window, &["pill"])?;
    let runtime = runtime.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        tauri::async_runtime::block_on(runtime.toggle_dictation())
    })
    .await
    .map_err(|e| format!("toggle dictation task panicked: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::require_window_access_for_label;

    #[test]
    fn require_window_access_rejects_unlisted_labels() {
        assert!(require_window_access_for_label("settings", &["main", "pill"]).is_err());
        assert!(require_window_access_for_label("pill", &["main", "pill"]).is_ok());
    }
}
