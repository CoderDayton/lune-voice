// SPDX-License-Identifier: GPL-3.0-or-later

use lune_core::config::PrivacyMode;
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

/// Configuration-level styling preflight: remote processing is permitted, and
/// a remote provider is selected and set up (endpoint key + model present).
/// Returns the user-facing reason when not. Applying a pending pipeline reload
/// is handled separately by `AppRuntime::ensure_styling_llm_ready`.
async fn check_styling_llm_configured() -> Result<(), String> {
    run_persist("check styling llm readiness", || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let settings = lune_persist::SettingsStore::new(paths.settings_file())
            .load()
            .map_err(|e| format!("load settings: {e}"))?;
        // Checked first: `LocalOnly` overrides a fully configured provider, so
        // reporting "ready" here and then degrading to the raw transcript at
        // dictation time is the one failure the user cannot diagnose. Styling
        // is remote-only, so this setting alone decides whether it can run.
        if matches!(settings.privacy_mode, PrivacyMode::LocalOnly) {
            return Err(
                "Privacy mode is set to local-only, so styling cannot send text to \
                        a remote service. Allow remote processing in Settings → Privacy."
                    .to_string(),
            );
        }
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
        activation_mode: snapshot.settings.core.activation_mode,
        pill_visibility: snapshot.settings.core.pill_visibility,
        pill_locked: snapshot.settings.core.pill_locked,
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
    runtime.mark_settings_changed().await;
    // Hot-reload the hotkey listener — trigger key, modifiers, cancel key, and
    // activation mode all apply immediately, no restart required. The
    // controller covers every case: a live native backend re-targets in place,
    // and a startup with no native listener (unresolvable key name) retries the
    // start here, so a binding repaired in the UI takes effect at once.
    {
        use tauri::Manager;
        if let Some(paths) = lune_persist::AppPaths::resolve() {
            // TOML read + parse is blocking disk I/O; keep it off the Tokio
            // worker running this command, like every other I/O in this file.
            let loaded = run_persist("reload settings for hotkey update", move || {
                lune_persist::SettingsStore::new(paths.settings_file())
                    .load()
                    .map_err(|e| e.to_string())
            })
            .await;
            match loaded {
                Ok(s) => {
                    if let Some(live) = runtime
                        .app_handle()
                        .try_state::<std::sync::Arc<crate::hotkey::HotkeyController>>()
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
    // Reconcile long-lived audio capture with the freshly-saved input device,
    // taken from the returned snapshot rather than `inner.settings`, which
    // stays stale until the next pipeline rebuild.
    runtime
        .sync_capture_device(Some(snapshot.settings.core.input_device.clone()))
        .await;
    // Same staleness reason as the mic reconcile above: the idle-unload
    // watchdog reads `inner.settings`, so hand it the freshly-saved value
    // instead of leaving the change to take effect a dictation late.
    runtime
        .sync_idle_unload(snapshot.settings.core.model_idle_unload_minutes)
        .await;
    // Apply sound-cue settings live — they're played from the backend and must
    // not lag a dictation behind a toggle/volume change.
    runtime
        .sync_sound_cues(
            snapshot.settings.core.sound_cues_enabled,
            snapshot.settings.core.sound_cue_volume,
        )
        .await;
    // Push pill overlay settings (geometry/placement/lock/pinning) into the
    // live window BEFORE create/destroy below, so a window created by
    // set_pill_enabled is built from the fresh config. A geometry change
    // rebuilds an existing window; `show_after` re-maps it per the same
    // visibility rule reconcile uses.
    {
        let vis =
            crate::persistence::pill_visibility_from_dto(snapshot.settings.core.pill_visibility);
        let status = runtime.service_status().await;
        let show_after = crate::pill::should_show_pill(vis, &status);
        if let Some(paths) = lune_persist::AppPaths::resolve() {
            let loaded = run_persist("reload settings for pill update", move || {
                lune_persist::SettingsStore::new(paths.settings_file())
                    .load()
                    .map_err(|e| e.to_string())
            })
            .await;
            match loaded {
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
        snapshot.settings.core.pill_visibility != crate::dto::PillVisibilityDto::Never,
    );
    // Map/unmap the pill to the new visibility right now, against the live
    // status — otherwise a switch to "Always" (or away from it) wouldn't take
    // effect until the next dictation republished the status. Runs after
    // set_pill_enabled so the window exists (Never -> enabled) before we show it.
    runtime
        .reconcile_pill_visibility(crate::persistence::pill_visibility_from_dto(
            snapshot.settings.core.pill_visibility,
        ))
        .await;
    // Notify the pill of the one setting it renders from. Everything else
    // (geometry, placement, visibility) is applied to the window itself above.
    if let Err(e) = runtime.app_handle().emit_to(
        "pill",
        "lune://pill-settings",
        serde_json::json!({ "pillLocked": snapshot.settings.core.pill_locked }),
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
    runtime.mark_config_changed().await;
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
pub async fn delete_sessions(
    window: WebviewWindow,
    runtime: State<'_, AppRuntime>,
    session_ids: Vec<String>,
) -> Result<WorkspaceSnapshot, String> {
    require_window_access(&window, &["main"])?;
    let mut snapshot = run_persist("delete sessions", move || {
        crate::persistence::delete_sessions(session_ids)
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
    runtime.mark_config_changed().await;
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
    runtime.mark_config_changed().await;
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
    runtime.mark_config_changed().await;
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
    runtime.mark_config_changed().await;
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
    runtime.mark_config_changed().await;
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
    runtime.mark_config_changed().await;
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
/// Whether every local model the *active* configuration needs is on disk.
///
/// This is the question first-run has to answer, and it is not the same as
/// "does any model exist": a remote ASR backend needs no local files at all,
/// while the local one needs exactly the Whisper size that is selected. The
/// welcome screen used to approximate it by looking for models flagged
/// `required` in [`check_model_status`] — a flag no spec sets, so the test was
/// vacuously true and setup could be skipped into an app that cannot transcribe.
///
/// Shares [`lune_persist::has_required_models`] with the runtime's own
/// readiness check, so the screen and the pipeline cannot disagree.
#[tauri::command]
pub async fn required_models_ready(window: WebviewWindow) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    run_persist("check required models", || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let store = lune_persist::SettingsStore::new(paths.settings_file());
        let settings = store.load().map_err(|e| format!("load settings: {e}"))?;
        Ok(lune_persist::has_required_models(
            &paths.models_dir(),
            &settings.asr_backend,
        ))
    })
    .await
}

/// The model download in flight, or `None`.
///
/// Global rather than per-window on purpose. The main webview is destroyed when
/// the window is closed and rebuilt when it reopens, so a download started by
/// one instance of the UI has to be visible to the next — otherwise closing the
/// window mid-setup looks like the download vanished, and reopening would start
/// a second copy of a transfer that is still running.
static ACTIVE_DOWNLOAD: std::sync::Mutex<Option<ActiveDownload>> = std::sync::Mutex::new(None);

/// The transfer in flight, plus a way to learn how it ends.
struct ActiveDownload {
    local_name: String,
    /// Terminal outcome, published once by [`release_download`].
    ///
    /// A second call for the same model waits on this instead of resolving
    /// straight away. The window that started the transfer may be gone — that is
    /// the whole point of the global slot — so its rejection goes nowhere, and a
    /// re-attached window that assumed success would sit in `downloading`
    /// forever waiting for a `done` event a failed transfer never emits.
    outcome: tokio::sync::broadcast::Sender<Result<(), String>>,
}

/// Set to ask the in-flight download to stop. Cleared when a download is claimed
/// so a cancel can never carry over into the next one.
static DOWNLOAD_CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A poisoned lock means a previous holder panicked while swapping a name. The
/// value is one `Option<ActiveDownload>` with no invariant across it, so
/// recovering it beats turning someone else's panic into ours.
fn active_download_slot() -> std::sync::MutexGuard<'static, Option<ActiveDownload>> {
    ACTIVE_DOWNLOAD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The result of asking for the download slot.
enum DownloadClaim {
    /// This call owns the transfer and must release it when it ends.
    Owned,
    /// The same model is already downloading. Starting a second transfer of the
    /// same files would only fight the first over one temp path, so this call
    /// waits on the receiver and reports whatever the running one does.
    Joined(tokio::sync::broadcast::Receiver<Result<(), String>>),
}

/// Claim the download slot for `local_name`, or attach to the transfer already
/// running for it.
fn claim_download(local_name: &str) -> Result<DownloadClaim, String> {
    let mut slot = active_download_slot();
    match slot.as_ref() {
        Some(active) if active.local_name == local_name => {
            // A pending cancel belongs to a request this call has just
            // superseded — the user pressed Cancel and then asked for the same
            // download again. Clearing it is what makes that work: otherwise the
            // transfer being attached to aborts on a flag nobody is waiting on
            // any more, and nothing is left downloading.
            DOWNLOAD_CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
            Ok(DownloadClaim::Joined(active.outcome.subscribe()))
        }
        Some(active) => Err(format!(
            "{} is already downloading — wait for it to finish",
            active.local_name
        )),
        None => {
            DOWNLOAD_CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
            let (outcome, _) = tokio::sync::broadcast::channel(1);
            *slot = Some(ActiveDownload {
                local_name: local_name.to_string(),
                outcome,
            });
            Ok(DownloadClaim::Owned)
        }
    }
}

/// Exact error text a cancelled download rejects with.
///
/// The frontend matches on it to tell "you pressed Cancel" apart from "the
/// download broke" — the two look identical from a rejected promise otherwise,
/// and showing a failure for something the user just asked for is wrong.
pub const DOWNLOAD_CANCELLED: &str = "model download cancelled";

/// Free the slot and tell anyone who attached how the transfer ended.
fn release_download(outcome: Result<(), String>) {
    let previous = active_download_slot().take();
    DOWNLOAD_CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
    if let Some(active) = previous {
        // A send with no receivers is the ordinary case — nobody re-attached —
        // and there is nothing to do about it either way.
        let _ = active.outcome.send(outcome);
    }
}

/// Which model is downloading right now, so a freshly-opened window can
/// re-attach to it instead of restarting setup from the first step.
#[tauri::command]
pub fn active_model_download(window: WebviewWindow) -> Result<Option<String>, String> {
    require_window_access(&window, &["main"])?;
    Ok(active_download_slot()
        .as_ref()
        .map(|active| active.local_name.clone()))
}

/// Ask the in-flight download to stop. Returns immediately; the transfer notices
/// within a chunk and its command call fails with a cancellation error.
///
/// A no-op when nothing is downloading, which is the honest answer for a button
/// the user pressed just as the last file finished.
#[tauri::command]
pub fn cancel_model_download(window: WebviewWindow) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    DOWNLOAD_CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
    Ok(())
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
    loop {
        match claim_download(&local_name)? {
            DownloadClaim::Owned => break,
            DownloadClaim::Joined(mut outcome) => match outcome.recv().await {
                // The transfer this call attached to was stopped by a cancel the
                // same call has already withdrawn (see `claim_download`). The
                // user asked for the download, so start it rather than reporting
                // their own superseded cancel back at them.
                Ok(Err(e)) if e == DOWNLOAD_CANCELLED => continue,
                Ok(result) => return result,
                // The owner released the slot between the claim and the send.
                // The transfer is over either way, and the caller settles it
                // from the model's state on disk.
                Err(_) => return Ok(()),
            },
        }
    }
    let app_handle = app.clone();
    let result = run_persist("download model", move || {
        let paths = lune_persist::AppPaths::resolve().ok_or("resolve application paths")?;
        let models_dir = paths.models_dir();
        let settings = lune_persist::SettingsStore::new(paths.settings_file())
            .load()
            .map_err(|e| format!("load settings: {e}"))?;
        tracing::info!(local_name, "starting model download");
        match lune_persist::download_model(&models_dir, &local_name, &DOWNLOAD_CANCEL, |progress| {
            let _ = app_handle.emit("lune://download-progress", &progress);
        }) {
            Ok(_) => {
                tracing::info!(local_name, "download complete");
                Ok((models_dir, settings.asr_backend))
            }
            Err(lune_persist::PersistError::ModelDownloadCancelled) => {
                tracing::info!(local_name, "download cancelled");
                Err(DOWNLOAD_CANCELLED.to_string())
            }
            Err(e) => {
                tracing::error!(local_name, error = %e, "download failed");
                Err(format!("download failed: {e}"))
            }
        }
    })
    .await;

    // Released whichever way the transfer ended, so a failure or a cancel never
    // leaves the slot claimed and every later download refused — and the same
    // outcome reaches any window that re-attached to this transfer.
    release_download(result.as_ref().map(|_| ()).map_err(String::clone));

    let (models_dir, asr_backend) = result?;
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

/// Whether this session can already type into other applications.
///
/// Setup asks this so it can request the permission up front. Left to itself the
/// runtime installs the udev rule lazily, on the first delivered keystroke —
/// which puts a polkit password prompt in the middle of the user's first
/// dictation, where it reads as a malfunction and where cancelling it loses the
/// utterance they just spoke.
#[tauri::command]
pub fn uinput_access_ready(window: WebviewWindow) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    #[cfg(unix)]
    {
        Ok(lune_voice::uinput_access_ready())
    }
    #[cfg(not(unix))]
    {
        Ok(true)
    }
}

/// Whether the Hold activation mode works right now.
///
/// `false` means the global-shortcut plugin is handling the hotkey, and it
/// can only toggle.
#[tauri::command]
pub fn hold_to_talk_available(window: WebviewWindow) -> Result<bool, String> {
    use tauri::Manager;

    require_window_access(&window, &["main"])?;
    Ok(window
        .app_handle()
        .try_state::<std::sync::Arc<crate::hotkey::HotkeyController>>()
        .is_some_and(|controller| controller.hold_available()))
}

/// Install the udev rule that lets this user synthesise keystrokes.
///
/// Raises a polkit prompt, hence `spawn_blocking`. Returns whether access works
/// afterwards rather than only whether the write succeeded, because udev takes a
/// moment to apply the new ACL and "the file exists" is not the question.
#[tauri::command]
pub async fn grant_uinput_access(window: WebviewWindow) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    #[cfg(unix)]
    {
        let granted = tauri::async_runtime::spawn_blocking(lune_voice::grant_uinput_access)
            .await
            .map_err(|e| format!("grant uinput access task: {e}"))?;
        // The grab that failed at startup is possible now. Retry it here:
        // nothing else would, and the user has no reason to expect that
        // saving an unrelated setting is what makes their hotkey work.
        if matches!(granted, Ok(true)) {
            retry_native_hotkeys(&window);
        }
        granted
    }
    #[cfg(not(unix))]
    {
        Ok(true)
    }
}

/// Ask the hotkey controller to start the native listener now.
///
/// Called after a permission grant, which changes what the evdev backend can
/// do without changing any setting.
#[cfg(unix)]
fn retry_native_hotkeys(window: &WebviewWindow) {
    use tauri::Manager;

    let Some(controller) = window
        .app_handle()
        .try_state::<std::sync::Arc<crate::hotkey::HotkeyController>>()
    else {
        return;
    };
    // No defaults here: the listener is being started *with* this binding, so
    // falling back to `UserSettings::default()` would silently re-bind the
    // hotkey to the stock key and activation mode instead of the user's.
    // Leaving the global-shortcut fallback in place is the honest outcome.
    let Some(settings) = lune_persist::AppPaths::resolve().and_then(|paths| {
        lune_persist::SettingsStore::new(paths.settings_file())
            .load()
            .ok()
    }) else {
        tracing::warn!(
            "uinput access was granted but the settings could not be read; \
             not starting the native listener on default bindings"
        );
        return;
    };
    if controller.retry_native(&settings.hotkey, settings.activation_mode) {
        tracing::info!("native hotkey listener is live after the uinput grant");
    } else {
        tracing::warn!(
            "uinput access was granted but no keyboard could be grabbed; \
             global shortcuts remain in use"
        );
    }
}

/// Whether this app installed the `/dev/uinput` udev rule itself.
///
/// False on a .deb/.rpm install, where the rule ships with the package and the
/// package manager removes it — offering to delete it there would break `dpkg`'s
/// view of its own files.
#[tauri::command]
pub fn uinput_rule_installed(window: WebviewWindow) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    #[cfg(unix)]
    {
        Ok(lune_voice::uinput_rule_is_self_installed())
    }
    #[cfg(not(unix))]
    {
        Ok(false)
    }
}

/// Remove the udev rule this app installed.
///
/// The rule grants every process running as the logged-in user permanent write
/// access to `/dev/uinput` — enough to synthesise keystrokes into any
/// application — so it needs a way out that does not involve the user hunting
/// through `/etc` themselves. Raises a polkit prompt, hence `spawn_blocking`.
#[tauri::command]
pub async fn remove_uinput_rule(window: WebviewWindow) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    #[cfg(unix)]
    {
        tauri::async_runtime::spawn_blocking(lune_voice::remove_uinput_rule)
            .await
            .map_err(|e| format!("remove udev rule task: {e}"))?
    }
    #[cfg(not(unix))]
    {
        Ok(false)
    }
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
        lune_asr::gpu_backend::GpuPluginState::NeedsRestart => {
            (GpuBackendStateDto::NeedsRestart, Vec::new(), None)
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
        download_available: lune_persist::gpu_plugin::CURRENT
            .is_some_and(lune_persist::gpu_plugin::GpuPluginRelease::is_published),
    })
}

/// Archive size narrowed for the IPC boundary. Saturating rather than wrapping:
/// a wrong-but-large number degrades the progress bar, a wrapped one would show
/// a nonsensical download size.
fn archive_bytes() -> u32 {
    let bytes = lune_persist::gpu_plugin::CURRENT.map_or(0, |r| r.download_bytes);
    u32::try_from(bytes).unwrap_or(u32::MAX)
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
    // No release for this platform at all (macOS). Distinct from an unpublished
    // one, which `install` itself reports.
    let release =
        lune_persist::gpu_plugin::CURRENT.ok_or("There is no GPU backend for this platform.")?;
    let paths = lune_persist::AppPaths::resolve().ok_or("failed to resolve application paths")?;
    let dest = lune_asr::gpu_backend::downloaded_plugin_dir(paths.data_dir());

    lune_persist::gpu_plugin::install(release, &dest, |p| {
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
