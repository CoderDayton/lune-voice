// SPDX-License-Identifier: GPL-3.0-or-later

//! Update checking and installing, reported through the app's own UI.
//!
//! Every outcome a user can see is an [`UpdateStateDto`] the main window
//! renders in its own modal. Nothing here opens a native message box: those are
//! drawn by the desktop environment, share none of the app's typography or
//! theme, and on Linux can open behind the window that asked for them — a
//! "Check for updates" that answers into a box nobody notices is the silent
//! no-op this flow exists to avoid.
//!
//! The state is *stored*, not merely emitted. A launch check can find an update
//! while the main window is closed, and an event aimed at a window that does not
//! exist is lost, so the frontend reads [`update_state`] on mount and picks up
//! whatever is waiting.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::utils::config::BundleType;
use tauri::{AppHandle, Emitter, EventTarget, Manager, State, WebviewWindow};
use tauri_plugin_updater::{Update, UpdaterExt};

use crate::commands::require_window_access;
use crate::dto::{UpdatePhaseDto, UpdateStateDto};

/// Whether this build checks for updates at all.
///
/// Off while the release assets are unreachable: the configured endpoint is a
/// release asset on a private repo, and GitHub answers the plugin's
/// unauthenticated GET with a 404. Every check would therefore end in
/// [`UpdatePhaseDto::CheckFailed`], which reads to the user as a broken app
/// rather than as an absent feature. Turn this back on together with the GPU
/// plugin pin in `lune_persist::gpu_plugin` — both wait on the same thing.
const UPDATES_ENABLED: bool = false;

/// Shown when the user asks for a check while [`UPDATES_ENABLED`] is off.
const DISABLED_MESSAGE: &str =
    "Automatic updates are turned off in this build. Install new versions manually.";

/// Event the main window listens on for every transition of this flow.
const UPDATE_EVENT: &str = "lune://update";

/// Bytes that must arrive before the download re-reports progress.
///
/// The plugin calls back per chunk — thousands of times for a package this
/// size — and every emit crosses the IPC boundary and re-renders the modal. A
/// megabyte is finer than the progress bar can show and coarse enough that the
/// download is not paying for its own reporting.
const PROGRESS_EMIT_BYTES: u32 = 1024 * 1024;

/// Set while a check is in flight, so the tray item and the Settings button
/// cannot run two at once.
static CHECK_RUNNING: AtomicBool = AtomicBool::new(false);
/// Set while an install is in flight. Separate from the check flag: the check
/// has long finished by the time the user answers the offer.
static INSTALL_RUNNING: AtomicBool = AtomicBool::new(false);

/// The update a check found, plus the last state the UI was told about.
pub struct UpdateFlow {
    /// Held from the moment a check finds an update until it is installed or
    /// dismissed. Kept across a failed install so *Try again* has something to
    /// retry without a second round trip to the release server.
    pending: Mutex<Option<Update>>,
    state: Mutex<UpdateStateDto>,
}

impl UpdateFlow {
    pub fn new(current_version: String) -> Self {
        Self {
            pending: Mutex::new(None),
            state: Mutex::new(UpdateStateDto::phase(
                UpdatePhaseDto::Idle,
                &current_version,
            )),
        }
    }
}

/// A poisoned lock here means a previous holder panicked mid-update. The data
/// is a plain snapshot with no invariant spanning the two fields, so recovering
/// the value is strictly better than turning someone else's panic into ours.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Record a new state without telling the window about it.
///
/// Used for a launch check that has already announced itself by notification:
/// the offer is kept so opening the window later still surfaces it, but nothing
/// interrupts whatever the user is doing right now.
fn set_state(app: &AppHandle, next: UpdateStateDto) {
    *lock(&app.state::<UpdateFlow>().state) = next;
}

/// Record a new state and push it to the main window.
fn publish(app: &AppHandle, next: UpdateStateDto) {
    set_state(app, next.clone());
    // Addressed at the main window rather than broadcast: the pill overlay is a
    // second webview that has no update UI and no business being woken for one.
    let _ = app.emit_to(EventTarget::webview_window("main"), UPDATE_EVENT, &next);
}

/// How loudly a check reports a non-event.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CheckMode {
    /// The user asked, so answer — including "you are up to date" and failures.
    Requested,
    /// Nobody asked. Only an actual update is worth interrupting for; being
    /// current, or being offline at login, must stay silent.
    OnLaunch,
}

impl CheckMode {
    /// Whether a non-event — already current, or a check that failed outright —
    /// is worth telling the user about. Only if they asked: "no network yet" is
    /// the normal state a second after login.
    pub fn reports_non_events(self) -> bool {
        matches!(self, Self::Requested)
    }

    /// Whether a found update is announced by notification instead of by
    /// opening the modal. A user who just clicked *Check for Updates…* is
    /// waiting on an answer and gets the modal; nobody asked for the launch
    /// check, so it must not throw a dialog over whatever they are doing.
    pub fn announces_by_notification(self) -> bool {
        matches!(self, Self::OnLaunch)
    }
}

/// Claim the single-flight slot for a check. `false` means one is already running.
pub fn begin_check() -> bool {
    !CHECK_RUNNING.swap(true, Ordering::SeqCst)
}

pub fn end_check() {
    CHECK_RUNNING.store(false, Ordering::SeqCst);
}

/// Tray "Check for Updates…" and the launch check.
pub async fn guarded_update_check(app: AppHandle, mode: CheckMode) {
    if !begin_check() {
        tracing::debug!("update check already in progress — ignoring the extra request");
        return;
    }
    run_update_check(&app, mode).await;
    end_check();
}

/// Title and body for the "an update is waiting" notification.
///
/// Split out from delivery so it can be asserted without a notification daemon.
/// Both arguments are versions and either order reads plausibly, so getting
/// them the wrong way round would ship a notification that tells the user they
/// already have the version they are being offered.
pub fn update_notification_text(version: &str, current: &str) -> (String, String) {
    (
        format!("Lune Voice {version} is available"),
        format!("You're running {current}."),
    )
}

/// Announce an available update without stealing focus.
///
/// Cross-platform by construction: `tauri-plugin-notification` fans out to
/// libnotify on Linux, WinRT toasts on Windows and `NSUserNotification` on
/// macOS, and its desktop `permission_state` is unconditionally `Granted`, so
/// no permission handshake is needed off mobile.
///
/// It announces and nothing more. No desktop backend gives us a click callback
/// — the plugin only wires notification actions up on Android and iOS — but the
/// offer is not lost by that: it stays in [`UpdateFlow`], so the modal is
/// waiting the next time the window is opened.
///
/// **Delivery cannot be confirmed.** Every desktop platform hands the
/// notification to a spawned task and returns `Ok` immediately, discarding the
/// result (`tauri-plugin-notification`'s `desktop.rs`), so a session with no
/// notification daemon is indistinguishable here from a delivered one. `false`
/// therefore means only that the plugin rejected the call — which is worth
/// falling back to the in-app modal for, but is not the same as "the user saw
/// it".
///
/// Generic over the runtime so tests can post through a mock app.
pub fn notify_update_available<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    version: &str,
    current: &str,
) -> bool {
    use tauri_plugin_notification::NotificationExt;

    let (title, body) = update_notification_text(version, current);

    // The icon is deliberately left unset. The desktop backend then falls back
    // to the executable's name (`lune-voice`), which is exactly the hicolor
    // icon the .deb/.rpm and `scripts/dev/desktop-entry.sh` install — whereas a
    // hardcoded name or path would only be right on one platform.
    match app.notification().builder().title(title).body(body).show() {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(error = %e, "the notification plugin refused the update notice");
            false
        }
    }
}

/// Every outcome of a *requested* check reports back in the app's modal,
/// including "already current" and each failure: a menu item that silently does
/// nothing is indistinguishable from a broken one. Nothing is downloaded before
/// the user agrees.
///
/// All three Linux bundles are supported — the plugin swaps an AppImage in
/// place and installs a .deb/.rpm through `pkexec` (a graphical password
/// prompt) — as is the Windows NSIS installer, which the plugin runs directly.
/// A release binary with no package stamp — the portable archive, or one run
/// straight out of `target/` — is offered a manual download instead (see
/// [`installs_in_place`]).
///
/// The manifest key is left to the plugin: it asks for
/// `linux-x86_64-{deb,rpm,appimage}` matching how this install was packaged and
/// falls back to the bare `{os}-x86_64` key — which is what Windows uses, and
/// exactly the scheme `scripts/release/updater-manifest.sh` publishes, without
/// losing the fallback that pinning an explicit target would remove.
/// Point the check at a local manifest to exercise download + install for real
/// (`cgs updater-test`, which builds `--debug`).
///
/// Debug builds only. A signature is still checked against the pinned public
/// key, so this could never introduce an unsigned update — but in a shipped
/// binary it would let anyone who controls the environment choose which host the
/// update check talks to, and that is worth nothing to us and something to an
/// attacker.
///
/// `Err` carries the message the modal shows: a set-but-broken override is a
/// developer mistake worth naming, not something to silently fall back from.
#[cfg(debug_assertions)]
fn apply_endpoint_override(
    builder: tauri_plugin_updater::UpdaterBuilder,
) -> Result<tauri_plugin_updater::UpdaterBuilder, String> {
    let Ok(endpoint) = std::env::var("LUNE_UPDATER_ENDPOINT") else {
        return Ok(builder);
    };
    tracing::warn!(%endpoint, "updater endpoint overridden for testing");
    url::Url::parse(&endpoint)
        .map_err(|e| e.to_string())
        .and_then(|url| builder.endpoints(vec![url]).map_err(|e| e.to_string()))
        .map_err(|e| {
            format!("LUNE_UPDATER_ENDPOINT is not a usable update endpoint.\n\n{endpoint}\n\n{e}")
        })
}

/// Answer a check made while [`UPDATES_ENABLED`] is off.
///
/// A launch check nobody asked for stays silent, exactly as it does when the
/// network is down at login; a check the user clicked still gets an answer, in
/// the window that answer has to land in.
fn report_updates_disabled(app: &AppHandle, mode: CheckMode, current: &str) {
    if !mode.reports_non_events() {
        return;
    }
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || crate::create_or_focus_main_window(&handle));
    publish(
        app,
        UpdateStateDto {
            message: Some(DISABLED_MESSAGE.to_string()),
            // `Disabled`, not `CheckFailed`: nothing failed. The failure phase
            // heads the modal "Update check failed", paints the notice as an
            // error and offers "Try again" — which is exactly the "broken app"
            // impression this whole gate exists to avoid.
            ..UpdateStateDto::phase(UpdatePhaseDto::Disabled, current)
        },
    );
}

pub async fn run_update_check(app: &AppHandle, mode: CheckMode) {
    let current = app.package_info().version.to_string();

    if !UPDATES_ENABLED {
        report_updates_disabled(app, mode, &current);
        return;
    }

    if mode.reports_non_events() {
        // The user asked, so give the answer somewhere to land: the tray item
        // works with the main window closed, and a modal with no window to
        // render in would be exactly the silent no-op this flow avoids.
        //
        // GTK window creation has to happen on the main thread, and this runs
        // inside a spawned task.
        let handle = app.clone();
        let _ = app.run_on_main_thread(move || crate::create_or_focus_main_window(&handle));
        publish(
            app,
            UpdateStateDto::phase(UpdatePhaseDto::Checking, &current),
        );
    }

    let builder = app.updater_builder();

    #[cfg(debug_assertions)]
    let builder = match apply_endpoint_override(builder) {
        Ok(b) => b,
        Err(message) => {
            tracing::error!(%message, "endpoint override refused");
            if mode.reports_non_events() {
                publish(
                    app,
                    UpdateStateDto {
                        message: Some(message),
                        ..UpdateStateDto::phase(UpdatePhaseDto::CheckFailed, &current)
                    },
                );
            }
            return;
        }
    };

    let found = match builder.build() {
        Ok(updater) => updater.check().await,
        Err(e) => Err(e),
    };

    let update = match found {
        Ok(Some(update)) => update,
        Ok(None) => {
            if mode.reports_non_events() {
                publish(
                    app,
                    UpdateStateDto::phase(UpdatePhaseDto::UpToDate, &current),
                );
            }
            return;
        }
        Err(e) => {
            tracing::warn!(error = %e, "update check failed");
            // A launch check runs before anyone has asked for anything, and
            // "no network yet" is the normal case a second after login.
            if mode.reports_non_events() {
                publish(
                    app,
                    UpdateStateDto {
                        message: Some(format!("Could not check for updates.\n\n{e}")),
                        ..UpdateStateDto::phase(UpdatePhaseDto::CheckFailed, &current)
                    },
                );
            }
            return;
        }
    };

    announce_update(app, mode, update, &current);
}

/// Whether the updater can replace this copy in place.
///
/// The bundler stamps a binary with the package it went into. On Linux, an
/// unstamped release binary — the portable archive, whose stamp
/// `package-all.sh` clears, or one run straight out of `target/` — has no
/// package to replace, and the plugin would hand its download to the AppImage
/// installer. Debug builds keep the in-place path for `cgs updater-test`.
fn installs_in_place(bundle: Option<BundleType>, debug_build: bool) -> bool {
    debug_build || bundle.is_some() || !cfg!(target_os = "linux")
}

/// Store a found update and tell the user about it in the way `mode` calls for.
fn announce_update(app: &AppHandle, mode: CheckMode, update: Update, current: &str) {
    let version = update.version.clone();
    let notes = update.body.clone();

    // A copy that cannot be replaced keeps no `Update`: with nothing pending,
    // `install_pending_update` refuses rather than running an installer.
    let phase = if installs_in_place(
        tauri::utils::platform::bundle_type(),
        cfg!(debug_assertions),
    ) {
        *lock(&app.state::<UpdateFlow>().pending) = Some(update);
        UpdatePhaseDto::Available
    } else {
        UpdatePhaseDto::ManualDownload
    };

    let offer = UpdateStateDto {
        available_version: Some(version.clone()),
        notes,
        ..UpdateStateDto::phase(phase, current)
    };

    // A check nobody asked for announces by notification and stores the offer
    // without opening anything: twenty seconds into a session the user is
    // working in another window, and a modal thrown over that is far more
    // disruptive than the update is urgent. Opening the app later finds the
    // offer waiting. If the notification cannot even be handed over we still
    // owe them the news, so fall through to the modal.
    if mode.announces_by_notification() && notify_update_available(app, &version, current) {
        tracing::info!(%version, "update available — notified");
        set_state(app, offer);
        return;
    }

    publish(app, offer);
}

// ── Commands ─────────────────────────────────────────────────────────────

/// Settings' **Check for updates** and the sidebar quick action. Runs the same
/// flow as the tray item and returns immediately; the outcome arrives as
/// `lune://update`.
///
/// Returns `false` when a check was already in flight, so the caller can tell
/// "nothing started" apart from "started and still running" instead of leaving
/// a button spinning at an answer that will never come.
#[tauri::command]
pub async fn start_update_check(window: WebviewWindow, app: AppHandle) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    if !begin_check() {
        return Ok(false);
    }
    tauri::async_runtime::spawn(async move {
        run_update_check(&app, CheckMode::Requested).await;
        end_check();
    });
    Ok(true)
}

/// The current update state, for a window that just mounted.
///
/// This is what keeps a launch-time discovery from being lost: the event that
/// announced it was never sent (or was sent at a window that did not exist yet),
/// and this is where the offer was kept instead.
#[tauri::command]
pub async fn update_state(
    window: WebviewWindow,
    flow: State<'_, UpdateFlow>,
) -> Result<UpdateStateDto, String> {
    require_window_access(&window, &["main"])?;
    Ok(lock(&flow.state).clone())
}

/// Download and apply the update the last check found, then restart.
///
/// The offer is cloned rather than taken: a failed install leaves it in place
/// so *Try again* costs one download instead of another round trip to the
/// release server.
#[tauri::command]
pub async fn install_pending_update(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    require_window_access(&window, &["main"])?;

    let update = lock(&app.state::<UpdateFlow>().pending)
        .clone()
        .ok_or("no update is waiting to be installed")?;

    if INSTALL_RUNNING.swap(true, Ordering::SeqCst) {
        // Already running. Not an error: a double-click on *Install* is a user
        // asking once, and the modal is already showing the download.
        return Ok(());
    }

    let offer = Offer {
        current: app.package_info().version.to_string(),
        version: update.version.clone(),
        notes: update.body.clone(),
    };

    publish(&app, offer.progress(UpdatePhaseDto::Downloading, 0, 0));

    let result = run_install(&app, &update, &offer).await;
    INSTALL_RUNNING.store(false, Ordering::SeqCst);

    match result {
        Ok(()) => {
            tracing::info!(version = %offer.version, "update installed — restarting");
            app.restart();
        }
        Err(e) => {
            tracing::error!(error = %e, version = %offer.version, "update install failed");
            publish(
                &app,
                UpdateStateDto {
                    available_version: Some(offer.version),
                    message: Some(e.to_string()),
                    ..UpdateStateDto::phase(UpdatePhaseDto::InstallFailed, &offer.current)
                },
            );
            Ok(())
        }
    }
}

/// The three strings every state this install publishes has to repeat.
#[derive(Clone)]
struct Offer {
    current: String,
    version: String,
    notes: Option<String>,
}

impl Offer {
    fn progress(&self, phase: UpdatePhaseDto, downloaded: u32, total: u32) -> UpdateStateDto {
        UpdateStateDto {
            available_version: Some(self.version.clone()),
            notes: self.notes.clone(),
            downloaded_bytes: downloaded,
            total_bytes: total,
            ..UpdateStateDto::phase(phase, &self.current)
        }
    }
}

/// Drive the plugin's download + install, reporting byte progress as it goes.
async fn run_install(
    app: &AppHandle,
    update: &Update,
    offer: &Offer,
) -> tauri_plugin_updater::Result<()> {
    let chunk_app = app.clone();
    let chunk_offer = offer.clone();
    let finish_app = app.clone();
    let finish_offer = offer.clone();
    let mut downloaded: u32 = 0;
    let mut last_emitted: u32 = 0;

    update
        .download_and_install(
            move |chunk, total| {
                // Saturating throughout: the package is a few hundred MB, but a
                // server is free to claim any `Content-Length`, and that claim
                // must not be able to turn this callback into a panic.
                downloaded = downloaded.saturating_add(u32::try_from(chunk).unwrap_or(u32::MAX));
                let total = total
                    .map(|t| u32::try_from(t).unwrap_or(u32::MAX))
                    .unwrap_or(0);
                if downloaded.saturating_sub(last_emitted) >= PROGRESS_EMIT_BYTES {
                    last_emitted = downloaded;
                    publish(
                        &chunk_app,
                        chunk_offer.progress(UpdatePhaseDto::Downloading, downloaded, total),
                    );
                }
            },
            move || {
                // Applying a .deb/.rpm raises a polkit password prompt, so this
                // phase can sit for as long as the user takes to type it. Naming
                // the step is the difference between that and a progress bar
                // that appears to have frozen at 100%.
                publish(
                    &finish_app,
                    finish_offer.progress(UpdatePhaseDto::Installing, 0, 0),
                );
            },
        )
        .await
}

/// Close the modal: forget the offer and go quiet until the next check.
///
/// Dropping the `Update` is deliberate. Keeping it would mean *Later* silently
/// re-opening the same modal on the next window mount, which is nagging rather
/// than informing — the next check will find the same version again anyway.
#[tauri::command]
pub async fn dismiss_update(
    window: WebviewWindow,
    app: AppHandle,
    flow: State<'_, UpdateFlow>,
) -> Result<(), String> {
    require_window_access(&window, &["main"])?;
    *lock(&flow.pending) = None;
    let current = app.package_info().version.to_string();
    *lock(&flow.state) = UpdateStateDto::phase(UpdatePhaseDto::Idle, &current);
    Ok(())
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    /// The configured endpoint is a release asset on a private repo, which
    /// GitHub serves as a 404 to the unauthenticated request the plugin makes.
    /// Until that changes, every check would end in `CheckFailed`, so the flow
    /// is off rather than reliably broken.
    // Pinning a constant is the whole point here: this test is the thing that
    // fails, with the reason attached, when someone flips the flag back on.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn update_checks_are_disabled() {
        assert!(
            !UPDATES_ENABLED,
            "re-enable only once the release assets are anonymously reachable"
        );
    }

    /// The disabled notice lands verbatim in the update modal, so it must read
    /// as a sentence to a user and name no endpoint, repo or HTTP status.
    #[test]
    fn the_disabled_notice_is_user_facing() {
        let msg = DISABLED_MESSAGE;
        assert!(msg.ends_with('.'), "should be a sentence: {msg}");
        for leak in ["http", "github", "404", "endpoint", "repo"] {
            assert!(
                !msg.to_ascii_lowercase().contains(leak),
                "leaks {leak:?} at the user: {msg}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_check_cannot_start_while_one_is_running() {
        // The tray item stays clickable while a check runs, and Settings has a
        // button for the same thing. Two concurrent checks would race over one
        // `download_and_install` — or raise two `pkexec` prompts.
        assert!(begin_check(), "the first check should claim the slot");
        assert!(!begin_check(), "a second check must be refused");
        end_check();
        assert!(begin_check(), "the slot should be free once the check ends");
        end_check();
    }

    #[test]
    fn a_progress_state_keeps_naming_the_version_being_installed() {
        // The modal shows one heading and one version through download and
        // install alike; dropping it mid-flight would leave the user watching a
        // progress bar for an unnamed thing.
        let offer = Offer {
            current: "1.3.2".into(),
            version: "1.4.0".into(),
            notes: Some("fixes".into()),
        };

        let state = offer.progress(UpdatePhaseDto::Downloading, 512, 2048);

        assert_eq!(state.phase, UpdatePhaseDto::Downloading);
        assert_eq!(state.available_version.as_deref(), Some("1.4.0"));
        assert_eq!(state.current_version, "1.3.2");
        assert_eq!(state.notes.as_deref(), Some("fixes"));
        assert_eq!(state.downloaded_bytes, 512);
        assert_eq!(state.total_bytes, 2048);
    }

    /// A Linux copy with no package stamp — the portable archive, or a release
    /// binary run out of `target/` — has no package the updater can replace:
    /// the plugin would hand the download to its AppImage installer. It is
    /// offered a manual download instead. Debug builds keep the in-place path
    /// so `cgs updater-test` can exercise it.
    #[test]
    fn only_a_packaged_copy_installs_updates_in_place() {
        assert!(installs_in_place(Some(BundleType::Deb), false));
        assert!(installs_in_place(Some(BundleType::AppImage), false));
        assert!(installs_in_place(None, true));
        assert_eq!(
            installs_in_place(None, false),
            !cfg!(target_os = "linux"),
            "an unpackaged Linux release build must not install in place"
        );
    }

    #[test]
    fn a_plain_phase_carries_no_stale_offer() {
        // `..phase(..)` is spread under every richer state, so anything it left
        // populated would leak a previous check's version into the next answer.
        let state = UpdateStateDto::phase(UpdatePhaseDto::UpToDate, "1.3.2");

        assert_eq!(state.current_version, "1.3.2");
        assert!(state.available_version.is_none());
        assert!(state.notes.is_none());
        assert!(state.message.is_none());
        assert_eq!(state.downloaded_bytes, 0);
        assert_eq!(state.total_bytes, 0);
    }
}
