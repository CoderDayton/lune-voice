mod commands;
mod dto;
#[cfg(target_os = "linux")]
mod hotkey_linux;
#[cfg(test)]
mod notification_tests;
mod persistence;
mod pill;
#[cfg(target_os = "linux")]
mod pill_native;
#[cfg(target_os = "linux")]
mod pill_paint;
mod runtime;
#[cfg(test)]
mod tauri_integration_tests;
#[cfg(test)]
mod updater_tests;

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
#[cfg(not(target_os = "linux"))]
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

/// Set when the user explicitly requested a quit (tray "Quit" menu).
/// Without this flag, the ExitRequested handler can't tell intentional
/// quits apart from window-closes, and we'd block both.
static QUIT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Restore the user's IBus engine on a shutdown signal, then exit.
///
/// `IbusTyper`'s `Drop` is the normal restore path, but it does not run for a
/// `SIGTERM` — which is exactly how a session ends on logout or `systemctl
/// stop`. Without this, quitting that way while a commit is in flight leaves
/// IBus on our no-op engine and the user unable to type. The on-disk record
/// still covers `SIGKILL` and panics by recovering at the next launch; this
/// just avoids making the user wait for that.
#[cfg(target_os = "linux")]
const SHUTDOWN_RESTORE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

#[cfg(target_os = "linux")]
fn install_shutdown_signal_handler() {
    use tokio::signal::unix::{SignalKind, signal};

    tauri::async_runtime::spawn(async move {
        let (Ok(mut term), Ok(mut int)) = (
            signal(SignalKind::terminate()),
            signal(SignalKind::interrupt()),
        ) else {
            tracing::debug!("could not install shutdown signal handlers");
            return;
        };
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
        tracing::info!("shutdown signal received; restoring input method");
        // Blocking D-Bus work: keep it off the async worker, and bound it. The
        // call has no timeout of its own, so an ibus-daemon that accepts the
        // connection and stops answering would turn SIGTERM into a process that
        // never exits — the user's `kill` (or logout) would appear to do
        // nothing until systemd's stop timeout escalated to SIGKILL.
        let restore = tokio::task::spawn_blocking(lune_voice::restore_displaced_engine_now);
        if tokio::time::timeout(SHUTDOWN_RESTORE_TIMEOUT, restore)
            .await
            .is_err()
        {
            tracing::warn!("input-method restore timed out; exiting anyway");
        }
        std::process::exit(0);
    });
}

/// Frontend-invokable quit. Mirrors the tray "Quit" path: set the intent
/// flag so the ExitRequested handler honors the exit, then terminate.
///
/// Gated to the main window like every other command: Tauri's ACL does not
/// cover app-defined commands, so `require_window_access` is the only thing
/// keeping the pill webview from invoking this.
#[tauri::command]
fn quit_app(window: tauri::WebviewWindow, app: tauri::AppHandle) -> Result<(), String> {
    commands::require_window_access(&window, &["main"])?;
    QUIT_REQUESTED.store(true, Ordering::SeqCst);
    app.exit(0);
    Ok(())
}

/// Smallest usable window. Also the OS-enforced minimum and the floor for the
/// adaptive default, so the UI never opens below its designed layout width.
const MIN_WINDOW_W: f64 = 1280.0;
const MIN_WINDOW_H: f64 = 720.0;
/// Upper bound on the adaptive default — caps ultrawide and 4K monitors so the
/// window opens at a comfortable size instead of a sprawling one.
const MAX_WINDOW_W: f64 = 1600.0;
const MAX_WINDOW_H: f64 = 1100.0;
/// Fraction of the monitor the default window aims to fill (~70%, mid the
/// desired 60–75% range).
const WINDOW_SCREEN_FRACTION: f64 = 0.70;

/// Adaptive default window size: ~70% of the primary monitor's logical size,
/// clamped to `[MIN, MAX]` per axis. The width cap keeps ultrawides from
/// opening an absurdly wide window; the floor keeps small laptops at the UI's
/// designed minimum. Falls back to the min size if the monitor can't be read.
fn adaptive_window_size(app: &tauri::AppHandle) -> (f64, f64) {
    let Ok(Some(monitor)) = app.primary_monitor() else {
        return (MIN_WINDOW_W, MIN_WINDOW_H);
    };
    // Guard against a bogus scale (0 / NaN / inf) so the division can't yield a
    // non-finite size the builder would choke on.
    let raw_scale = monitor.scale_factor();
    let scale = if raw_scale.is_finite() && raw_scale > 0.0 {
        raw_scale
    } else {
        1.0
    };
    let size = monitor.size();
    let logical_w = size.width as f64 / scale;
    let logical_h = size.height as f64 / scale;
    let width = (logical_w * WINDOW_SCREEN_FRACTION).clamp(MIN_WINDOW_W, MAX_WINDOW_W);
    let height = (logical_h * WINDOW_SCREEN_FRACTION).clamp(MIN_WINDOW_H, MAX_WINDOW_H);
    (width, height)
}

/// At most one update check at a time: the menu item stays clickable while a
/// check runs, and two concurrent `download_and_install` calls would race over
/// the same binary (or raise two `pkexec` prompts).
static UPDATE_CHECK_RUNNING: AtomicBool = AtomicBool::new(false);

/// How long a launch check waits before touching the network, so startup and
/// an autostarted session's network setup both finish first.
const UPDATE_CHECK_LAUNCH_DELAY: std::time::Duration = std::time::Duration::from_secs(20);

/// How loudly a check reports a non-event.
#[derive(Clone, Copy, PartialEq, Debug)]
enum CheckMode {
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
    fn reports_non_events(self) -> bool {
        matches!(self, Self::Requested)
    }

    /// Whether a found update is announced by notification instead of by a
    /// dialog. A user who just clicked *Check for Updates…* is waiting on an
    /// answer and gets the dialog; nobody asked for the launch check, so it
    /// must not seize focus out from under whatever they are actually doing.
    fn announces_by_notification(self) -> bool {
        matches!(self, Self::OnLaunch)
    }
}

/// Tray "Check for Updates…" and the Settings button.
async fn check_for_updates(app: tauri::AppHandle) {
    guarded_update_check(app, CheckMode::Requested).await;
}

async fn guarded_update_check(app: tauri::AppHandle, mode: CheckMode) {
    if UPDATE_CHECK_RUNNING.swap(true, Ordering::SeqCst) {
        tracing::debug!("update check already in progress — ignoring the extra request");
        return;
    }
    run_update_check(&app, mode).await;
    UPDATE_CHECK_RUNNING.store(false, Ordering::SeqCst);
}

/// Title and body for the "an update is waiting" notification.
///
/// Split out from delivery so it can be asserted without a notification daemon.
/// Both arguments are versions and either order reads plausibly, so getting
/// them the wrong way round would ship a notification that tells the user they
/// already have the version they are being offered.
fn update_notification_text(version: &str, current: &str) -> (String, String) {
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
/// — the plugin only wires notification actions up on Android and iOS — but
/// spelling out where to click instead would be worse: notifications are a
/// glance, not a manual.
///
/// **Delivery cannot be confirmed.** Every desktop platform hands the
/// notification to a spawned task and returns `Ok` immediately, discarding the
/// result (`tauri-plugin-notification`'s `desktop.rs`), so a session with no
/// notification daemon is indistinguishable here from a delivered one. `false`
/// therefore means only that the plugin rejected the call — which is worth a
/// dialog fallback, but is not the same as "the user saw it".
///
/// Generic over the runtime so tests can post through a mock app.
fn notify_update_available<R: tauri::Runtime>(
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

/// Every outcome of a *requested* check reports back in a dialog, including
/// "already current" and each failure: a menu item that silently does nothing
/// is indistinguishable from a broken one. Nothing is downloaded before the
/// user agrees.
///
/// All three Linux bundles are supported — the plugin swaps an AppImage in
/// place and installs a .deb/.rpm through `pkexec` (a graphical password
/// prompt). A binary run straight out of `target/` has no package to replace,
/// so the install step fails and says so rather than pretending to work.
///
/// The manifest key is left to the plugin: it asks for
/// `linux-x86_64-{deb,rpm,appimage}` matching how this install was packaged and
/// falls back to the bare `linux-x86_64` key — exactly the per-format scheme
/// `scripts/release/updater-manifest.sh` publishes, but without losing the
/// fallback that pinning an explicit target would remove.
/// Show a message dialog without parking a Tokio worker.
///
/// `blocking_show()` blocks its calling thread until the dialog is dismissed —
/// arbitrarily long for the install prompt. `run_update_check` always runs
/// inside `tauri::async_runtime::spawn(...)`, so the raw call held a runtime
/// worker for the whole interaction. Mirrors `commands::export_transcript`'s
/// wrap of `blocking_save_file()`.
async fn show_dialog(dialog: tauri_plugin_dialog::MessageDialogBuilder<tauri::Wry>) -> bool {
    tauri::async_runtime::spawn_blocking(move || dialog.blocking_show())
        .await
        .unwrap_or(false)
}

async fn run_update_check(app: &tauri::AppHandle, mode: CheckMode) {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
    use tauri_plugin_updater::UpdaterExt;

    let current = app.package_info().version.to_string();

    #[cfg_attr(not(debug_assertions), allow(unused_mut))]
    let mut builder = app.updater_builder();

    // Point the check at a local manifest to exercise download + install for
    // real (`cgs updater-test`, which builds `--debug`).
    //
    // Debug builds only. A signature is still checked against the pinned public
    // key, so this could never introduce an unsigned update — but in a shipped
    // binary it would let anyone who controls the environment choose which host
    // the update check talks to, and that is worth nothing to us and something
    // to an attacker.
    #[cfg(debug_assertions)]
    if let Ok(endpoint) = std::env::var("LUNE_UPDATER_ENDPOINT") {
        let overridden = url::Url::parse(&endpoint)
            .map_err(|e| e.to_string())
            .and_then(|url| builder.endpoints(vec![url]).map_err(|e| e.to_string()));
        match overridden {
            Ok(b) => {
                tracing::warn!(%endpoint, "updater endpoint overridden for testing");
                builder = b;
            }
            Err(e) => {
                tracing::error!(error = %e, %endpoint, "endpoint override refused");
                show_dialog(
                    app.dialog()
                        .message(format!(
                            "LUNE_UPDATER_ENDPOINT is not a usable update endpoint.\n\n{endpoint}\n\n{e}"
                        ))
                        .title("Update check failed")
                        .kind(MessageDialogKind::Error),
                )
                .await;
                return;
            }
        }
    }
    let found = match builder.build() {
        Ok(updater) => updater.check().await,
        Err(e) => Err(e),
    };

    let update = match found {
        Ok(Some(update)) => update,
        Ok(None) => {
            if mode.reports_non_events() {
                show_dialog(
                    app.dialog()
                        .message(format!("Lune Voice {current} is the latest version."))
                        .title("No updates available")
                        .kind(MessageDialogKind::Info),
                )
                .await;
            }
            return;
        }
        Err(e) => {
            tracing::warn!(error = %e, "update check failed");
            // A launch check runs before anyone has asked for anything, and
            // "no network yet" is the normal case a second after login.
            if mode.reports_non_events() {
                show_dialog(
                    app.dialog()
                        .message(format!("Could not check for updates.\n\n{e}"))
                        .title("Update check failed")
                        .kind(MessageDialogKind::Error),
                )
                .await;
            }
            return;
        }
    };

    let version = update.version.clone();

    // A check nobody asked for gets a notification, not a modal. Twenty seconds
    // into a session the user is working in another window, and a dialog that
    // seizes focus there is far more disruptive than the update is urgent. If
    // the notification cannot be delivered we still owe them the news, so fall
    // through to the dialog.
    if mode.announces_by_notification() && notify_update_available(app, &version, &current) {
        tracing::info!(%version, "update available — notified");
        return;
    }

    let install = show_dialog(
        app.dialog()
            .message(format!(
                "Lune Voice {version} is available. You have {current}.\n\n\
                 Installing replaces the app and restarts it."
            ))
            .title("Update available")
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Install and restart".into(),
                "Later".into(),
            )),
    )
    .await;

    if !install {
        return;
    }

    match update.download_and_install(|_, _| {}, || {}).await {
        Ok(()) => {
            tracing::info!(%version, "update installed — restarting");
            app.restart();
        }
        Err(e) => {
            tracing::error!(error = %e, %version, "update install failed");
            show_dialog(
                app.dialog()
                    .message(format!("Could not install {version}.\n\n{e}"))
                    .title("Update failed")
                    .kind(MessageDialogKind::Error),
            )
            .await;
        }
    }
}

/// Bring the main window to front, creating it on demand if it doesn't exist.
/// We don't keep the window alive while minimized — closing destroys the
/// webview to free its memory, and the next call here rebuilds it.
fn create_or_focus_main_window(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
        return;
    }
    let (width, height) = adaptive_window_size(app);
    if let Err(e) = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .title("Lune Voice")
        .inner_size(width, height)
        .min_inner_size(MIN_WINDOW_W, MIN_WINDOW_H)
        .resizable(true)
        .decorations(true)
        .center()
        .build()
    {
        tracing::error!(error = %e, "failed to create main window");
    }
}

pub fn run() {
    // On a Wayland session whose compositor lacks wlr-layer-shell (GNOME/Mutter,
    // most KDE), a native Wayland window can neither position itself nor stay
    // always-on-top — both of which the dictation pill needs. Run the whole app
    // under XWayland there so set_position + _NET_WM_STATE_ABOVE work again.
    // COSMIC and other wlroots compositors keep native Wayland + layer-shell.
    // Must happen before the Tauri builder touches GTK/GDK.
    #[cfg(target_os = "linux")]
    if std::env::var_os("GDK_BACKEND").is_none() && lune_voice::should_use_xwayland_fallback() {
        // SAFETY: single-threaded here — main() has only initialised tracing
        // before calling run(); no other thread reads the environment yet.
        unsafe { std::env::set_var("GDK_BACKEND", "x11") };
        tracing::info!(
            "Wayland without layer-shell detected — running under XWayland \
             (GDK_BACKEND=x11) so the pill can position and stay on top"
        );
    }

    // Register ggml's compute backends before anything can build a
    // WhisperContext: ggml picks a device from whatever is in the registry at
    // context-creation time, so a late load would silently leave the app on
    // the CPU. Bundled CPU plugins and the optional downloaded CUDA plugin are
    // both picked up here.
    if let Some(paths) = lune_persist::AppPaths::resolve() {
        let devices = lune_asr::gpu_backend::load_plugins(paths.data_dir());
        tracing::info!(?devices, "ggml compute backends available");
    } else {
        tracing::warn!("could not resolve application paths; GPU plugins not loaded");
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            create_or_focus_main_window(app);
        }))
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            #[cfg(target_os = "linux")]
            install_shutdown_signal_handler();
            let runtime =
                tauri::async_runtime::block_on(runtime::AppRuntime::new(app.handle().clone()))
                    .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            let start_minimized = tauri::async_runtime::block_on(runtime.start_minimized());
            let pill_vis = tauri::async_runtime::block_on(runtime.pill_visibility());
            if tauri::async_runtime::block_on(runtime.check_updates_on_launch()) {
                // Off the startup path entirely: a slow or unreachable endpoint
                // must never delay the window or the hotkey. The wait also
                // gives an autostarted session time to get its network up,
                // since "offline one second after login" is the common case.
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    tokio::time::sleep(UPDATE_CHECK_LAUNCH_DELAY).await;
                    guarded_update_check(handle, CheckMode::OnLaunch).await;
                });
            }
            // Seed the pill's geometry/placement config and saved position (if
            // the user dragged it) before the overlay's first configure/show,
            // so the window is created with the user's size/anchor and reopens
            // where they left it.
            pill::seed_pill_config(tauri::async_runtime::block_on(
                runtime.pill_overlay_config(),
            ));
            if let Some(pos) = tauri::async_runtime::block_on(runtime.pill_position()) {
                pill::set_saved_position(pos.x, pos.y);
            }
            app.manage(runtime);

            // Build the main window only when not starting minimized. While
            // minimized the webview isn't created at all — the tray icon
            // (registered below) is the way to bring it up, which lazily
            // builds it via create_or_focus_main_window.
            if !start_minimized {
                create_or_focus_main_window(app.handle());
            }

            // Pill overlay window — created hidden UNLESS the user disabled it
            // (pill_visibility = Never), so "Never" spawns no webview process at
            // all. A later visibility change recreates/destroys it live via
            // save_workspace_settings → pill::set_pill_enabled. Dictation status
            // changes show/hide it (runtime::publish_status → pill.rs).
            if pill_vis != lune_core::config::PillVisibility::Never {
                pill::create_pill_window(app.handle());
                // Reconcile initial visibility against the startup status
                // ("ready"): "Always" shows the idle pill now; "WhenDictating"
                // waits for the first dictation, which publish_status handles.
                if pill::should_show_pill(pill_vis, "ready") {
                    pill::show_pill_direct(app.handle());
                }
            }

            // ── Global hotkeys ─────────────────────────────────────────────
            // Linux: evdev grab works on both X11 and Wayland.
            // Other platforms: tauri-plugin-global-shortcut.
            #[cfg(target_os = "linux")]
            spawn_evdev_hotkeys(app)?;
            #[cfg(not(target_os = "linux"))]
            register_hotkeys(app)?;

            // ── System tray ──────────────────────────────────────────────
            let show_i = MenuItem::with_id(app, "show", "Open Lune Voice", true, None::<&str>)?;
            let toggle_i = MenuItem::with_id(app, "toggle", "Start Dictation", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let updates_i = MenuItem::with_id(
                app,
                "check_updates",
                "Check for Updates…",
                true,
                None::<&str>,
            )?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &toggle_i, &sep, &updates_i, &quit_i])?;

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
                        create_or_focus_main_window(app);
                    }
                    "toggle" => {
                        let rt = app.state::<runtime::AppRuntime>();
                        let rt = rt.inner().clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = rt.toggle_dictation().await;
                        });
                    }
                    "check_updates" => {
                        tauri::async_runtime::spawn(check_for_updates(app.clone()));
                    }
                    "quit" => {
                        QUIT_REQUESTED.store(true, Ordering::SeqCst);
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
                        create_or_focus_main_window(app);
                    }
                })
                .build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::app_version,
            commands::start_update_check,
            commands::load_workspace_snapshot,
            commands::load_pill_snapshot,
            commands::save_workspace_settings,
            commands::create_dictionary_entry,
            commands::delete_session,
            commands::list_history_page,
            commands::delete_dictionary_entry,
            commands::update_dictionary_entry,
            commands::save_snippet,
            commands::delete_snippet,
            commands::export_transcript,
            commands::create_custom_style,
            commands::update_custom_style,
            commands::delete_custom_style,
            commands::generate_style_examples,
            commands::subscribe_session_stream,
            commands::detach_session_stream,
            commands::start_dictation,
            commands::start_trigger_capture,
            commands::stop_dictation,
            commands::cancel_dictation,
            commands::toggle_dictation,
            pill::nudge_pill,
            pill::save_pill_position,
            pill::reset_pill_position,
            pill::list_monitors,
            commands::list_audio_input_devices,
            commands::check_model_status,
            commands::download_model_cmd,
            commands::delete_model_cmd,
            commands::initialize_pipeline,
            commands::gpu_backend_status,
            commands::install_gpu_backend,
            quit_app
        ])
        .build(tauri::generate_context!())
        .unwrap_or_else(|e| {
            // Both sinks: stderr for a terminal launch, tracing so a desktop
            // launch leaves the reason in the log file (see `main.rs`).
            tracing::error!(error = %e, "fatal: tauri application failed to start");
            eprintln!("fatal: tauri application exited with error: {e}");
            std::process::exit(1);
        })
        .run(|_app, event| {
            // Closing the main window destroys it to free webview memory.
            // Without this, Tauri would also exit the app. We keep the
            // process alive so the tray icon, hotkeys, and dictation
            // runtime continue working — the user can rebuild the window
            // from the tray. Tray "Quit" sets QUIT_REQUESTED so its
            // explicit app.exit(0) is honored.
            if let tauri::RunEvent::ExitRequested { api, .. } = event
                && !QUIT_REQUESTED.load(Ordering::SeqCst)
            {
                api.prevent_exit();
            }
        });
}

#[cfg(not(target_os = "linux"))]
fn register_hotkeys(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // Read hotkey config directly — can't use block_on here since we're
    // already inside one from setup(). Just use the defaults; the runtime
    // has the same settings loaded but we can't await inside setup.
    let settings = lune_core::config::UserSettings::default();
    if let Some(paths) = lune_persist::AppPaths::resolve() {
        let store = lune_persist::SettingsStore::new(paths.settings_file());
        if let Ok(s) = store.load() {
            return register_hotkeys_with(app, &s);
        }
    }
    register_hotkeys_with(app, &settings)
}

#[cfg(not(target_os = "linux"))]
fn register_hotkeys_with(
    app: &tauri::App,
    settings: &lune_core::config::UserSettings,
) -> Result<(), Box<dyn std::error::Error>> {
    let toggle_str = format_hotkey(&settings.hotkey.toggle_dictation);
    let cancel_str = format_hotkey(&settings.hotkey.cancel_dictation);

    let toggle: Shortcut = toggle_str
        .parse()
        .map_err(|e| format!("parse toggle hotkey: {e}"))?;
    let cancel: Shortcut = cancel_str
        .parse()
        .map_err(|e| format!("parse cancel hotkey: {e}"))?;

    app.global_shortcut()
        .on_shortcut(toggle, move |app, _shortcut, event| {
            if event.state == ShortcutState::Pressed {
                let rt = app.state::<runtime::AppRuntime>();
                let rt = rt.inner().clone();
                tauri::async_runtime::spawn(async move {
                    let _ = rt.toggle_dictation().await;
                });
            }
        })?;

    app.global_shortcut()
        .on_shortcut(cancel, move |app, _shortcut, event| {
            if event.state == ShortcutState::Pressed {
                let rt = app.state::<runtime::AppRuntime>();
                let rt = rt.inner().clone();
                tauri::async_runtime::spawn(async move {
                    let _ = rt.cancel_dictation().await;
                });
            }
        })?;

    tracing::info!(toggle = %toggle_str, cancel = %cancel_str, "global hotkeys registered");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn format_hotkey(combo: &lune_core::config::KeyCombo) -> String {
    let mut parts: Vec<&str> = combo
        .modifiers
        .iter()
        .map(|m| match m {
            lune_core::config::Modifier::Ctrl => "ctrl",
            lune_core::config::Modifier::Shift => "shift",
            lune_core::config::Modifier::Alt => "alt",
            lune_core::config::Modifier::Meta => "super",
            _ => "unknown",
        })
        .collect();
    parts.push(&combo.key);
    parts.join("+")
}

#[cfg(target_os = "linux")]
fn spawn_evdev_hotkeys(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let settings = {
        let mut s = lune_core::config::UserSettings::default();
        if let Some(paths) = lune_persist::AppPaths::resolve() {
            let store = lune_persist::SettingsStore::new(paths.settings_file());
            if let Ok(loaded) = store.load() {
                s = loaded;
            }
        }
        s
    };

    let (sink, mut intents) = hotkey_linux::intent_channel();

    tracing::info!(
        toggle = ?settings.hotkey.toggle_dictation,
        cancel = ?settings.hotkey.cancel_dictation,
        mode = ?settings.activation_mode,
        "spawning evdev hotkey listener"
    );

    // Live config — managed in Tauri state so settings saves can hot-reload
    // every hotkey setting (trigger, modifiers, cancel, activation mode)
    // without restarting the listener thread.
    let Some(live) =
        hotkey_linux::LiveHotkeyConfig::new(&settings.hotkey, settings.activation_mode)
    else {
        tracing::warn!("evdev hotkey listener not started (unknown key name in config)");
        return Ok(());
    };
    let live = std::sync::Arc::new(live);
    app.manage(std::sync::Arc::clone(&live));

    match hotkey_linux::spawn(live, sink) {
        Some(_handle) => {
            // Thread handle intentionally leaked — runs for app lifetime.
            tracing::info!("evdev hotkey listener thread started");
        }
        None => {
            tracing::warn!("evdev hotkey listener thread failed to spawn");
            return Ok(());
        }
    }

    let rt = app.state::<runtime::AppRuntime>().inner().clone();
    tauri::async_runtime::spawn(async move {
        while let Some(action) = intents.recv().await {
            match action {
                hotkey_linux::HotkeyAction::Toggle => {
                    tracing::info!(action = "toggle", "hotkey pressed");
                    let _ = rt.toggle_dictation().await;
                }
                hotkey_linux::HotkeyAction::Start => {
                    tracing::info!(action = "start", "hotkey pressed");
                    let _ = rt.start_dictation().await;
                }
                hotkey_linux::HotkeyAction::Stop => {
                    tracing::info!(action = "stop", "hotkey pressed");
                    let _ = rt.stop_and_wait_dictation().await;
                }
                hotkey_linux::HotkeyAction::Cancel => {
                    if rt.is_active().await {
                        tracing::info!(action = "cancel", "hotkey pressed");
                        let _ = rt.cancel_dictation().await;
                    }
                }
            }
        }
        tracing::error!("hotkey channel closed — evdev listener has exited, hotkeys are dead");
    });

    tracing::info!("evdev hotkey listener spawned");
    Ok(())
}
