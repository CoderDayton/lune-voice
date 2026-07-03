use tauri::{Emitter, Runtime, State, WebviewWindow, ipc::Channel};

use crate::dto::{
    CreateCustomStyleInput, CreateDictionaryEntryInput, GpuSupportDto, HistoryPageDto,
    PillSnapshotDto, SaveWorkspaceSettingsInput, SessionStreamEvent, UpdateCustomStyleInput,
    UpdateDictionaryEntryInput, WorkspaceSnapshot,
};
use crate::runtime::AppRuntime;

/// Runs a blocking operation on the Tauri async runtime, handling errors uniformly.
///
/// This helper eliminates the boilerplate of spawn_blocking + map_err + double-?.
/// The closure should return Result<T, String>.
async fn run_persist<F, T>(label: &str, f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| format!("{label}: {e}"))?
}

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

/// Configuration-level styling preflight: provider selected and set up
/// (local model downloaded / remote key+model present). Returns the
/// user-facing reason when not. Initialization (model actually loaded) is
/// validated separately by `AppRuntime::ensure_styling_llm_ready`.
async fn check_styling_llm_configured() -> Result<(), String> {
    run_persist("check styling llm readiness", || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let settings = lune_persist::SettingsStore::new(paths.settings_file())
            .load()
            .map_err(|e| format!("load settings: {e}"))?;
        lune_persist::styling_llm_readiness(&paths.models_dir(), &settings.llm_backend)
            .map_err(|u| u.user_message().to_string())
    })
    .await
}

fn validate_local_name(local_name: &str) -> Result<(), String> {
    if local_name.contains('/')
        || local_name.contains('\\')
        || local_name.contains("..")
        || local_name.contains('\0')
    {
        Err("invalid model name: path fragments are not allowed".into())
    } else {
        Ok(())
    }
}

#[tauri::command]
pub async fn load_workspace_snapshot(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("load workspace snapshot", || {
        crate::persistence::load_workspace_snapshot()
    })
    .await?;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn load_pill_snapshot(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<PillSnapshotDto, String> {
    require_window_access(&window, &["pill"])?;
    let mut snapshot = run_persist("load pill snapshot", || {
        crate::persistence::load_workspace_snapshot()
    })
    .await?;
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
    input: SaveWorkspaceSettingsInput,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("save workspace settings", move || {
        crate::persistence::save_workspace_settings(input)
    })
    .await?;
    runtime.mark_reload_required().await;
    // Hot-reload the evdev hotkey listener — trigger key, modifiers, cancel
    // key, and activation mode all apply immediately, no restart required.
    #[cfg(target_os = "linux")]
    {
        use tauri::Manager;
        if let Some(paths) = lune_persist::AppPaths::resolve() {
            let store = lune_persist::SettingsStore::new(paths.settings_file());
            match store.load() {
                Ok(s) => {
                    if let Some(live) = runtime
                        .app_handle()
                        .try_state::<std::sync::Arc<crate::hotkey_linux::LiveHotkeyConfig>>()
                    {
                        if let Err(e) = live.update(&s.hotkey, s.activation_mode) {
                            tracing::warn!(error = %e, "hotkey hot-reload rejected");
                        }
                    }
                }
                Err(e) => tracing::warn!(error = %e, "settings reload for hotkey update failed"),
            }
        }
    }
    // Reconcile long-lived audio capture with the freshly-saved mic_prewarm and
    // input-device settings. Both are taken from the returned snapshot rather
    // than `inner.settings`, which stays stale until the next pipeline rebuild.
    runtime
        .sync_mic_prewarm(
            Some(snapshot.settings.mic_prewarm),
            Some(snapshot.settings.input_device.clone()),
        )
        .await;
    // Apply sound-cue settings live — they're played from the backend and must
    // not lag a dictation behind a toggle/volume change.
    runtime
        .sync_sound_cues(
            snapshot.settings.sound_cues_enabled,
            snapshot.settings.sound_cue_volume,
        )
        .await;
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
    let mut snapshot = run_persist("create dictionary entry", move || {
        crate::persistence::create_dictionary_entry(input)
    })
    .await?;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn delete_session(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    session_id: String,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("delete session", move || {
        crate::persistence::delete_session(session_id)
    })
    .await?;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn list_history_page(
    window: WebviewWindow,
    page: u32,
    page_size: u32,
    search: Option<String>,
) -> Result<HistoryPageDto, String> {
    require_window_access(&window, &["main"])?;
    run_persist("list history page", move || {
        crate::persistence::list_history_page(page, page_size, search)
    })
    .await
}

#[tauri::command]
pub async fn delete_dictionary_entry(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    entry_id: i32,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("delete dictionary entry", move || {
        crate::persistence::delete_dictionary_entry(entry_id)
    })
    .await?;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn update_dictionary_entry(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    input: UpdateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("update dictionary entry", move || {
        crate::persistence::update_dictionary_entry(input)
    })
    .await?;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn create_custom_style(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    input: CreateCustomStyleInput,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("create custom style", move || {
        crate::persistence::create_custom_style(input)
    })
    .await?;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn update_custom_style(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    input: UpdateCustomStyleInput,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("update custom style", move || {
        crate::persistence::update_custom_style(input)
    })
    .await?;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn delete_custom_style(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    style_id: String,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("delete custom style", move || {
        crate::persistence::delete_custom_style(style_id)
    })
    .await?;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn generate_style_examples(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    style_id: String,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    // Preflight: refuse to (re)generate unless the styling provider — and, for
    // the local engine, its model — is set up. Returns a user-facing message.
    check_styling_llm_configured().await?;
    // Then make sure the engine is actually initialized (pending pipeline
    // reload applied, local model loaded into memory) — a configured-but-cold
    // engine would otherwise fail generation with "local LLM not ready".
    runtime.ensure_styling_llm_ready().await?;
    let llm = runtime.llm_service().await;
    let (name, instructions) = run_persist("load custom style for examples", {
        let sid = style_id.clone();
        move || {
            let paths = lune_persist::AppPaths::resolve().ok_or("failed to resolve paths")?;
            let mut db = lune_persist::Database::open(&paths.database_file())
                .map_err(|e| format!("open database: {e}"))?;
            db.get_custom_style(&sid)
                .map_err(|e| format!("load custom style: {e}"))?
                .map(|s| (s.preset.name, s.preset.style_instructions))
                .ok_or_else(|| format!("custom style not found: {sid}"))
        }
    })
    .await?;
    let pairs = lune_llm::few_shot::generate_few_shot_examples(llm.as_ref(), &name, &instructions)
        .await
        .map_err(|e| e.to_string())?;
    // The generation call warmed the local model; give it the same idle TTL a
    // dictation gets so a styles-page visit doesn't pin VRAM indefinitely.
    runtime.schedule_local_llm_unload().await;
    let mut snapshot = run_persist("save style few-shot pairs", move || {
        crate::persistence::save_style_few_shot_pairs(&style_id, pairs)
    })
    .await?;
    runtime.mark_reload_required().await;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

/// Validate the styling engine end-to-end for the Styles page: provider
/// configured AND — for the local engine — the model loaded into memory and
/// able to serve. Blocks until the load completes (seconds on first call), so
/// the UI can gate style creation and the Regenerate button on a definitive
/// answer. Returns the user-facing reason on failure.
#[tauri::command]
pub async fn warm_up_styling_llm(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    check_styling_llm_configured().await?;
    runtime.ensure_styling_llm_ready().await
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

/// Enumerate the microphone input devices cpal currently reports, by name, for
/// the settings picker. An empty list means only the system default is offered.
#[tauri::command]
pub fn list_audio_input_devices(window: WebviewWindow) -> Result<Vec<String>, String> {
    require_window_access(&window, &["main"])?;
    Ok(lune_audio::list_input_devices())
}

#[tauri::command]
pub async fn start_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<String, String> {
    require_window_access(&window, &["main", "pill"])?;
    runtime.inner().clone().start_dictation().await
}

#[tauri::command]
pub async fn stop_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<(), String> {
    require_window_access(&window, &["main", "pill"])?;
    runtime.inner().clone().stop_dictation().await
}

#[tauri::command]
pub async fn cancel_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<(), String> {
    require_window_access(&window, &["main", "pill"])?;
    runtime.inner().clone().cancel_dictation().await
}

#[tauri::command]
pub async fn toggle_dictation(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<Option<String>, String> {
    require_window_access(&window, &["main", "pill"])?;
    runtime.inner().clone().toggle_dictation().await
}

#[tauri::command]
pub async fn check_model_status(
    window: WebviewWindow,
) -> Result<Vec<lune_persist::ModelStatus>, String> {
    require_window_access(&window, &["main"])?;
    run_persist("check model status", || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let store = lune_persist::SettingsStore::new(paths.settings_file());
        let settings = store.load().map_err(|e| format!("load settings: {e}"))?;
        Ok(lune_persist::check_model_status(
            &paths.models_dir(),
            &settings.asr_backend,
            &settings.llm_backend,
        ))
    })
    .await
}
#[tauri::command]
pub async fn download_model_cmd(
    window: WebviewWindow,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    local_name: String,
) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    validate_local_name(&local_name)?;
    let app_handle = app.clone();
    let (models_dir, asr_backend, llm_backend) = run_persist("download model", move || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let models_dir = paths.models_dir();
        let settings = lune_persist::SettingsStore::new(paths.settings_file())
            .load()
            .map_err(|e| format!("load settings: {e}"))?;
        tracing::info!(local_name, "starting model download");
        match lune_persist::download_model(&models_dir, &local_name, |progress| {
            let _ = app_handle.emit("lune://download-progress", &progress);
        }) {
            Ok(_) => {
                tracing::info!(local_name, "download complete");
                Ok((models_dir, settings.asr_backend, settings.llm_backend))
            }
            Err(e) => {
                tracing::error!(local_name, error = %e, "download failed");
                Err(format!("download failed: {e}"))
            }
        }
    })
    .await?;

    if lune_persist::has_required_models(&models_dir, &asr_backend, &llm_backend) {
        runtime.eager_rebuild().await.ok();
    }
    Ok(())
}

#[tauri::command]
pub async fn delete_model_cmd(window: WebviewWindow, local_name: String) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    validate_local_name(&local_name)?;
    run_persist("delete model", move || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        lune_persist::delete_model(&paths.models_dir(), &local_name)
            .map_err(|e| format!("delete model: {e}"))?;
        Ok(())
    })
    .await?;
    Ok(())
}

/// Reports whether GPU (CUDA) acceleration was compiled into this build.
///
/// GPU support is decided at build time: the native whisper.cpp/llama.cpp
/// backends only link CUDA when built with the `whisper-cuda`/`llama-cuda`
/// features. At runtime "is the GPU usable" therefore reduces to "was a CUDA
/// feature compiled in", which the onboarding flow uses to offer an
/// acceleration opt-in.
#[tauri::command]
pub fn gpu_support(window: WebviewWindow) -> Result<GpuSupportDto, String> {
    require_window_access(&window, &["main"])?;
    Ok(GpuSupportDto {
        compiled_in: cfg!(feature = "whisper-cuda") || cfg!(feature = "llama-cuda"),
    })
}

#[tauri::command]
pub async fn initialize_pipeline(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    runtime.eager_rebuild().await
}

#[cfg(test)]
mod tests {
    use super::{require_window_access_for_label, validate_local_name};

    #[test]
    fn require_window_access_rejects_unlisted_labels() {
        assert!(require_window_access_for_label("settings", &["main", "pill"]).is_err());
        assert!(require_window_access_for_label("main", &["main", "pill"]).is_ok());
        assert!(require_window_access_for_label("pill", &["main", "pill"]).is_ok());
    }

    #[test]
    fn validate_local_name_rejects_path_fragments() {
        assert!(validate_local_name("whisper-small").is_ok());
        assert!(validate_local_name("../whisper-small").is_err());
        assert!(validate_local_name("models/whisper-small").is_err());
        assert!(validate_local_name("models\\whisper-small").is_err());
        assert!(validate_local_name("whisper\0small").is_err());
    }
}
