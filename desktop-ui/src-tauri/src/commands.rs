use tauri::{AppHandle, Emitter, Runtime, State, WebviewWindow, ipc::Channel};
use tauri_plugin_dialog::DialogExt;

use crate::dto::{
    CreateCustomStyleInput, CreateDictionaryEntryInput, GpuBackendStateDto, GpuBackendStatusDto,
    GpuInstallProgressDto, HistoryPageDto, PillSnapshotDto, SaveSnippetInput,
    SaveWorkspaceSettingsInput, SessionStreamEvent, UpdateCustomStyleInput,
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

/// Fallback save-dialog file name, used when the frontend's suggestion is not
/// a usable bare file name.
const DEFAULT_TRANSCRIPT_FILENAME: &str = "lune-transcript.txt";
/// Save-dialog file-type filter for exported transcripts.
const TRANSCRIPT_FILTER_NAME: &str = "Text";
const TRANSCRIPT_EXTENSIONS: [&str; 1] = ["txt"];

/// Export a transcript to a user-chosen file. Returns `false` when the user
/// cancels the dialog.
///
/// The native save dialog runs *here*, in Rust, and the chosen path is never
/// round-tripped through the webview: the frontend supplies only a suggested
/// file name. That is what makes the write safe — a path accepted from the
/// frontend would be an arbitrary-file-write primitive with attacker-
/// influenced contents (`~/.bashrc`, `~/.config/autostart/*.desktop`) no
/// matter how it were validated, because nothing could tie it back to the
/// path the user actually picked.
#[tauri::command]
pub async fn export_transcript(
    app: AppHandle,
    window: WebviewWindow,
    transcript: String,
    default_name: String,
    title: String,
) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    // The suggestion and title are frontend-supplied (the title is localized
    // there). Keep only the suggestion's final component: a path fragment here
    // would seed the dialog somewhere the user didn't choose.
    let suggested = std::path::Path::new(&default_name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(DEFAULT_TRANSCRIPT_FILENAME)
        .to_owned();

    // `blocking_save_file` must not run on the main thread — it drives the
    // dialog there and waits for the reply.
    let picked = run_persist("transcript save dialog", move || {
        Ok(app
            .dialog()
            .file()
            .set_title(title)
            .set_file_name(suggested)
            .add_filter(TRANSCRIPT_FILTER_NAME, &TRANSCRIPT_EXTENSIONS)
            .blocking_save_file())
    })
    .await?;

    let Some(picked) = picked else {
        return Ok(false);
    };
    let path = picked
        .into_path()
        .map_err(|e| format!("export path is not a file path: {e}"))?;
    run_persist("export transcript", move || {
        std::fs::write(&path, transcript.as_bytes()).map_err(|e| format!("write transcript: {e}"))
    })
    .await?;
    Ok(true)
}

/// Configuration-level styling preflight: a remote provider is selected and
/// set up (endpoint key + model present). Returns the user-facing reason when
/// not. Applying a pending pipeline reload is handled separately by
/// `AppRuntime::ensure_styling_llm_ready`.
async fn check_styling_llm_configured() -> Result<(), String> {
    run_persist("check styling llm readiness", || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let settings = lune_persist::SettingsStore::new(paths.settings_file())
            .load()
            .map_err(|e| format!("load settings: {e}"))?;
        lune_persist::styling_llm_readiness(&settings.llm_backend)
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

/// Version this build reports — the same string the updater compares the
/// release against, so what Settings shows is what the check acts on.
#[tauri::command]
pub async fn app_version(window: WebviewWindow, app: tauri::AppHandle) -> Result<String, String> {
    require_window_access(&window, &["main"])?;
    Ok(app.package_info().version.to_string())
}

/// Settings' **Check for updates**. Runs the same flow as the tray item,
/// dialogs and all, and returns immediately — the single-flight guard around
/// that flow means the button and the tray menu cannot race each other.
#[tauri::command]
pub async fn start_update_check(
    window: WebviewWindow,
    app: tauri::AppHandle,
) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    tauri::async_runtime::spawn(crate::check_for_updates(app));
    Ok(())
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
        pill_locked: snapshot.settings.pill_locked,
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
    // Push pill overlay settings (geometry/placement/lock/pinning) into the
    // live window BEFORE create/destroy below, so a window created by
    // set_pill_enabled is built from the fresh config. A geometry change
    // rebuilds an existing window; `show_after` re-maps it per the same
    // visibility rule reconcile uses.
    {
        let vis = crate::persistence::pill_visibility_from_dto(snapshot.settings.pill_visibility);
        let status = runtime.service_status().await;
        let show_after = crate::pill::should_show_pill(vis, &status);
        if let Some(paths) = lune_persist::AppPaths::resolve() {
            match lune_persist::SettingsStore::new(paths.settings_file()).load() {
                Ok(s) => crate::pill::apply_pill_settings(runtime.app_handle(), &s, show_after),
                Err(e) => tracing::warn!(error = %e, "settings reload for pill update failed"),
            }
        }
    }
    // Create or destroy the pill overlay window to match the new visibility.
    // "Never" must free the webview process, not merely hide it; switching back
    // recreates it without an app restart. Runs before the settings emit below —
    // a freshly created pill reads current settings from its snapshot on load.
    crate::pill::set_pill_enabled(
        runtime.app_handle(),
        snapshot.settings.pill_visibility != crate::dto::PillVisibilityDto::Never,
    );
    // Map/unmap the pill to the new visibility right now, against the live
    // status — otherwise a switch to "Always" (or away from it) wouldn't take
    // effect until the next dictation republished the status. Runs after
    // set_pill_enabled so the window exists (Never -> enabled) before we show it.
    runtime
        .reconcile_pill_visibility(crate::persistence::pill_visibility_from_dto(
            snapshot.settings.pill_visibility,
        ))
        .await;
    // Notify the pill of the one setting it renders from. Everything else
    // (geometry, placement, visibility) is applied to the window itself above.
    if let Err(e) = runtime.app_handle().emit_to(
        "pill",
        "lune://pill-settings",
        serde_json::json!({ "pillLocked": snapshot.settings.pill_locked }),
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

/// Create or update a voice snippet. No pipeline reload: snippets are read
/// fresh from the database on every utterance.
#[tauri::command]
pub async fn save_snippet(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    input: SaveSnippetInput,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("save snippet", move || {
        crate::persistence::save_snippet(input)
    })
    .await?;
    snapshot.service_status = runtime.service_status().await;
    Ok(snapshot)
}

#[tauri::command]
pub async fn delete_snippet(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    snippet_id: String,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("delete snippet", move || {
        crate::persistence::delete_snippet(snippet_id)
    })
    .await?;
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
    // Preflight: refuse to (re)generate unless the remote styling provider is
    // set up. Returns a user-facing message.
    check_styling_llm_configured().await?;
    // Then make sure the live pipeline reflects the saved settings — a pending
    // reload would otherwise send the request to the previous engine.
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
    let mut snapshot = run_persist("save style few-shot pairs", move || {
        crate::persistence::save_style_few_shot_pairs(&style_id, pairs)
    })
    .await?;
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

/// Start a capture-only session: transcribes speech and streams the raw text
/// back as a `CaptureText` event instead of typing it anywhere. Used by the
/// snippet trigger recorder.
#[tauri::command]
pub async fn start_trigger_capture(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
) -> Result<String, String> {
    require_window_access(&window, &["main"])?;
    runtime.start_trigger_capture().await
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
    let (models_dir, asr_backend) = run_persist("download model", move || {
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
                Ok((models_dir, settings.asr_backend))
            }
            Err(e) => {
                tracing::error!(local_name, error = %e, "download failed");
                Err(format!("download failed: {e}"))
            }
        }
    })
    .await?;

    if lune_persist::has_required_models(&models_dir, &asr_backend) {
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

/// Runtime GPU status: driver present, plugin installed, device registered.
///
/// This can change on the same binary — the CUDA backend is a downloadable
/// plugin, so the answer depends on what the user has installed rather than on
/// how the app was compiled. Both the onboarding acceleration card and the
/// Settings GPU section read it.
#[tauri::command]
pub fn gpu_backend_status(window: WebviewWindow) -> Result<GpuBackendStatusDto, String> {
    require_window_access(&window, &["main"])?;
    current_gpu_status()
}

/// The one place a [`GpuBackendStatusDto`] is built, so a status read and the
/// value returned by an install can never disagree about the same machine.
fn current_gpu_status() -> Result<GpuBackendStatusDto, String> {
    let paths = lune_persist::AppPaths::resolve().ok_or("failed to resolve application paths")?;

    let (state, devices, reason) = match lune_asr::gpu_backend::state(paths.data_dir()) {
        lune_asr::gpu_backend::GpuPluginState::Unsupported => {
            (GpuBackendStateDto::Unsupported, Vec::new(), None)
        }
        lune_asr::gpu_backend::GpuPluginState::DriverMissing => {
            (GpuBackendStateDto::DriverMissing, Vec::new(), None)
        }
        lune_asr::gpu_backend::GpuPluginState::NotInstalled => {
            (GpuBackendStateDto::NotInstalled, Vec::new(), None)
        }
        lune_asr::gpu_backend::GpuPluginState::Ready { devices } => {
            (GpuBackendStateDto::Ready, devices, None)
        }
        lune_asr::gpu_backend::GpuPluginState::Failed { reason } => {
            (GpuBackendStateDto::Failed, Vec::new(), Some(reason))
        }
    };

    Ok(GpuBackendStatusDto {
        state,
        devices,
        reason,
        download_bytes: archive_bytes(),
        download_available: lune_persist::gpu_plugin::CUDA_LINUX_X86_64.is_published(),
    })
}

/// Archive size narrowed for the IPC boundary. Saturating rather than wrapping:
/// a wrong-but-large number degrades the progress bar, a wrapped one would show
/// a nonsensical download size.
fn archive_bytes() -> u32 {
    u32::try_from(lune_persist::gpu_plugin::CUDA_LINUX_X86_64.download_bytes).unwrap_or(u32::MAX)
}

/// Download and install the CUDA backend plugin.
///
/// Refuses when no NVIDIA driver is present: the driver is not redistributable,
/// so the download would install libraries that cannot load. Progress is
/// reported to the caller's channel as the archive streams.
#[tauri::command]
pub async fn install_gpu_backend(
    window: WebviewWindow,
    on_progress: Channel<GpuInstallProgressDto>,
) -> Result<GpuBackendStatusDto, String> {
    require_window_access(&window, &["main"])?;
    if !lune_asr::gpu_backend::nvidia_driver_present() {
        return Err(
            "No NVIDIA driver was found. Install the driver for your GPU first — it is not \
             something Lune Voice can download for you."
                .to_owned(),
        );
    }
    let paths = lune_persist::AppPaths::resolve().ok_or("failed to resolve application paths")?;
    let dest = lune_asr::gpu_backend::downloaded_plugin_dir(paths.data_dir());

    lune_persist::gpu_plugin::install(&lune_persist::gpu_plugin::CUDA_LINUX_X86_64, &dest, |p| {
        let _ = on_progress.send(GpuInstallProgressDto {
            downloaded_bytes: u32::try_from(p.downloaded_bytes).unwrap_or(u32::MAX),
            total_bytes: u32::try_from(p.total_bytes).unwrap_or(u32::MAX),
            done: p.done,
        });
    })
    .await
    .map_err(|e| e.to_string())?;

    // Re-read rather than assert: the freshly installed plugin cannot register
    // in this process (ggml's backend registry is built once at startup and has
    // no unload), and `gpu_backend::state` is what knows that, reporting the
    // restart-required reason. Returning a hand-rolled `not-installed` here
    // would put the download button back in front of the user.
    current_gpu_status()
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
