// SPDX-License-Identifier: GPL-3.0-or-later

mod commands;
mod dto;
mod feedback;
mod hotkey;
#[cfg(target_os = "linux")]
mod hotkey_linux;
// Compiled everywhere: the activation-mode state machine inside is portable,
// and building it on Linux is what keeps its tests running in CI. Off Windows
// nothing but those tests calls into it, which is not dead code — it is the
// point.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
mod hotkey_windows;
mod limits;
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
mod update;
#[cfg(test)]
mod updater_tests;
#[cfg(test)]
mod window_tests;

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, WebviewWindowBuilder};
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

/// Adaptive default window size: ~70% of a monitor's logical size, clamped to
/// `[MIN, MAX]` per axis. The width cap keeps ultrawides from opening an
/// absurdly wide window; the floor keeps small laptops at the UI's designed
/// minimum.
///
/// Prefers the primary monitor, then falls back to the largest one available.
/// That second step is not a corner case: GDK only has a notion of a primary
/// monitor on its X11 backend, so under native Wayland `primary_monitor()` is
/// *always* empty and going straight to the floor would open a 1280x720 window
/// on a 3440x1440 display. Only a machine that reports no monitors at all gets
/// the minimum.
fn adaptive_window_size(app: &tauri::AppHandle) -> (f64, f64) {
    let Some(monitor) = app
        .primary_monitor()
        .ok()
        .flatten()
        .or_else(|| largest_monitor(app))
    else {
        return (MIN_WINDOW_W, MIN_WINDOW_H);
    };
    let (logical_w, logical_h) = monitor_logical_size(&monitor);
    window_size_for(logical_w, logical_h)
}

/// The monitor with the most logical area, or `None` when none can be read.
/// Area rather than width so a tall secondary display can't win over a wider
/// primary one just by being listed first.
fn largest_monitor(app: &tauri::AppHandle) -> Option<tauri::Monitor> {
    app.available_monitors().ok()?.into_iter().max_by(|a, b| {
        let (aw, ah) = monitor_logical_size(a);
        let (bw, bh) = monitor_logical_size(b);
        (aw * ah).total_cmp(&(bw * bh))
    })
}

/// A monitor's logical (scale-corrected) size. Guards against a bogus scale
/// (0 / NaN / inf) so the division can't yield a non-finite size the window
/// builder would choke on.
fn monitor_logical_size(monitor: &tauri::Monitor) -> (f64, f64) {
    let raw_scale = monitor.scale_factor();
    let scale = if raw_scale.is_finite() && raw_scale > 0.0 {
        raw_scale
    } else {
        1.0
    };
    let size = monitor.size();
    (size.width as f64 / scale, size.height as f64 / scale)
}

/// The clamping itself, split out so it can be checked without a monitor.
///
/// `f64::clamp` propagates NaN, and this result goes straight into the window
/// builder, so a non-finite axis is folded onto the floor instead of passed on.
fn window_size_for(logical_w: f64, logical_h: f64) -> (f64, f64) {
    let axis = |logical: f64, min: f64, max: f64| {
        if logical.is_finite() {
            (logical * WINDOW_SCREEN_FRACTION).clamp(min, max)
        } else {
            min
        }
    };
    (
        axis(logical_w, MIN_WINDOW_W, MAX_WINDOW_W),
        axis(logical_h, MIN_WINDOW_H, MAX_WINDOW_H),
    )
}

/// How long a launch check waits before touching the network, so startup and
/// an autostarted session's network setup both finish first.
const UPDATE_CHECK_LAUNCH_DELAY: std::time::Duration = std::time::Duration::from_secs(20);

/// Bring the main window to front, creating it on demand if it doesn't exist.
/// We don't keep the window alive while minimized — closing destroys the
/// webview to free its memory, and the next call here rebuilds it.
fn create_or_focus_main_window(app: &tauri::AppHandle) {
    open_main_window(app, None);
}

/// Put the newest dictation's text on the clipboard.
fn copy_latest_dictation() {
    match persistence::latest_transcript() {
        Ok(Some(text)) => {
            if let Err(e) = lune_voice::copy_text(&text) {
                tracing::warn!(error = %e, "copy last dictation failed");
            }
        }
        Ok(None) => tracing::info!("no dictation to copy yet"),
        Err(e) => tracing::warn!(error = %e, "copy last dictation failed"),
    }
}

/// Show the main window, building it if needed, on `route` when one is given.
///
/// The router follows the browser location. A window that exists is moved with
/// a history push and a `popstate`; one that does not exist yet starts at the
/// route because the location is set before its scripts run.
fn open_main_window(app: &tauri::AppHandle, route: Option<&str>) {
    if let Some(w) = app.get_webview_window("main") {
        if let Some(route) = route {
            let _ = w.eval(format!(
                "history.pushState(null, '', '{route}'); \
                 dispatchEvent(new PopStateEvent('popstate'));"
            ));
        }
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
        return;
    }
    // The window's properties live in `tauri.conf.json` under `app.windows`,
    // declared with `"create": false` so Tauri does not build it at startup.
    // Only the size is decided here, because it adapts to the monitor.
    let Some(config) = app
        .config()
        .app
        .windows
        .iter()
        .find(|w| w.label == "main")
        .cloned()
    else {
        tracing::error!("no `main` window is declared in tauri.conf.json");
        return;
    };
    let (width, height) = adaptive_window_size(app);
    match WebviewWindowBuilder::from_config(app, &config) {
        Ok(builder) => {
            let builder = builder.inner_size(width, height);
            let builder = match route {
                Some(route) => builder
                    .initialization_script(format!("history.replaceState(null, '', '{route}');")),
                None => builder,
            };
            if let Err(e) = builder.build() {
                tracing::error!(error = %e, "failed to create main window");
            }
        }
        Err(e) => tracing::error!(error = %e, "failed to read the main window config"),
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
        // SAFETY: nothing else reads the environment at this point. main() has
        // only initialised tracing before calling run(), and the X probe inside
        // the call above is the sole other thread this process has started — it
        // finishes its `DISPLAY`/`XAUTHORITY` reads before answering, and only
        // an answer of `true` reaches this line. Any future `set_var` here must
        // re-check that, because the probe thread is deliberately abandoned
        // when it stalls.
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
    // Before the load, not after: ggml reports a failed plugin `dlopen` through
    // its own log callback, and until this is installed that goes to raw
    // stderr — invisible on a GUI launch, and unrecoverable even with
    // `RUST_LOG=info`.
    lune_asr::install_logging_hooks();
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
            // Seeded with the running version so the update modal can say what
            // the user has even before a check has said what exists.
            app.manage(update::UpdateFlow::new(
                app.package_info().version.to_string(),
            ));
            if tauri::async_runtime::block_on(runtime.check_updates_on_launch()) {
                // Off the startup path entirely: a slow or unreachable endpoint
                // must never delay the window or the hotkey. The wait also
                // gives an autostarted session time to get its network up,
                // since "offline one second after login" is the common case.
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    tokio::time::sleep(UPDATE_CHECK_LAUNCH_DELAY).await;
                    update::guarded_update_check(handle, update::CheckMode::OnLaunch).await;
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
            // Under WebDriver automation there is no tray for a test to click,
            // and a session attaches to a webview — so the window has to exist
            // regardless of the preference. `tauri-driver` sets this.
            let automated = std::env::var("TAURI_WEBVIEW_AUTOMATION").as_deref() == Ok("true");
            if !start_minimized || automated {
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
            spawn_hotkeys(app)?;

            // ── System tray ──────────────────────────────────────────────
            let show_i = MenuItem::with_id(app, "show", "Open Lune Voice", true, None::<&str>)?;
            let settings_i =
                MenuItem::with_id(app, "settings", "Open Settings", true, None::<&str>)?;
            let toggle_i = MenuItem::with_id(app, "toggle", "Start Dictation", true, None::<&str>)?;
            let copy_i = MenuItem::with_id(
                app,
                "copy_latest",
                "Copy Last Dictation",
                true,
                None::<&str>,
            )?;
            let sep = PredefinedMenuItem::separator(app)?;
            let updates_i = MenuItem::with_id(
                app,
                "check_updates",
                "Check for Updates…",
                true,
                None::<&str>,
            )?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(
                app,
                &[
                    &show_i,
                    &settings_i,
                    &toggle_i,
                    &copy_i,
                    &sep,
                    &updates_i,
                    &quit_i,
                ],
            )?;

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
                    "settings" => {
                        open_main_window(app, Some("/settings"));
                    }
                    "copy_latest" => {
                        tauri::async_runtime::spawn_blocking(copy_latest_dictation);
                    }
                    "toggle" => {
                        let rt = app.state::<runtime::AppRuntime>();
                        let rt = rt.inner().clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = rt.toggle_dictation().await;
                        });
                    }
                    "check_updates" => {
                        tauri::async_runtime::spawn(update::guarded_update_check(
                            app.clone(),
                            update::CheckMode::Requested,
                        ));
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
            update::start_update_check,
            update::update_state,
            update::install_pending_update,
            update::dismiss_update,
            commands::load_workspace_snapshot,
            commands::load_pill_snapshot,
            commands::save_workspace_settings,
            commands::create_dictionary_entry,
            commands::delete_session,
            commands::delete_sessions,
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
            commands::required_models_ready,
            commands::download_model_cmd,
            commands::delete_model_cmd,
            commands::gpu_backend_status,
            commands::uinput_access_ready,
            commands::grant_uinput_access,
            commands::hold_to_talk_available,
            commands::uinput_rule_installed,
            commands::remove_uinput_rule,
            commands::active_model_download,
            commands::cancel_model_download,
            commands::install_gpu_backend,
            feedback::feedback_available,
            feedback::feedback_cooldown_remaining,
            feedback::submit_feedback,
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

/// Load the persisted settings, or fall back to defaults.
///
/// Read directly rather than through the runtime: `setup` is already inside a
/// `block_on`, so there is nothing here that can await.
fn hotkey_settings() -> lune_core::config::UserSettings {
    lune_persist::AppPaths::resolve()
        .and_then(|paths| {
            lune_persist::SettingsStore::new(paths.settings_file())
                .load()
                .ok()
        })
        .unwrap_or_default()
}

/// Start global hotkeys on whichever backend this platform has.
///
/// A native backend (`hotkey::start`) is preferred everywhere it exists: it is
/// the only kind that can see a key *release*, and so the only kind that can
/// serve push-to-talk. `None` means this platform has none, and the
/// global-shortcut plugin takes over with toggle-only semantics.
fn spawn_hotkeys(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let settings = hotkey_settings();
    let (sink, mut intents) = hotkey::intent_channel();

    tracing::info!(
        toggle = ?settings.hotkey.toggle_dictation,
        cancel = ?settings.hotkey.cancel_dictation,
        mode = ?settings.activation_mode,
        "starting global hotkey listener"
    );

    let native = hotkey::start(&settings.hotkey, settings.activation_mode, sink.clone());
    // The pair the plugin actually holds, so a failed rebind can restore it.
    let mut plugin_bound = None;
    match &native {
        Some(backend) => {
            tracing::info!(backend = backend.name(), "native hotkey listener started")
        }
        None => {
            // Either this platform has no native backend, or the configured key
            // name did not resolve. Neither is fatal: an app that refuses to
            // start because a hotkey is misspelled leaves the user no way in to
            // fix it, and the controller retries the native start on the next
            // settings save.
            tracing::info!("no native hotkey backend; using the global-shortcut plugin");
            match register_hotkeys_with(app.handle(), &settings.hotkey) {
                Ok(()) => plugin_bound = Some(settings.hotkey.clone()),
                Err(e) => {
                    tracing::warn!(error = %e, "global shortcut registration failed; hotkeys are off")
                }
            }
        }
    }

    // Managed so a settings save can hot-reload the binding, the modifiers and
    // the activation mode without restarting the app — whichever listener is
    // live, and even when none is yet.
    app.manage(std::sync::Arc::new(hotkey::HotkeyController::new(
        native,
        sink,
        Box::new(PluginShortcuts {
            app: app.handle().clone(),
            bound: std::sync::Mutex::new(plugin_bound),
        }),
    )));

    let rt = app.state::<runtime::AppRuntime>().inner().clone();
    tauri::async_runtime::spawn(async move {
        while let Some(action) = intents.recv().await {
            dispatch_hotkey(&rt, action).await;
        }
        tracing::error!("hotkey channel closed — the listener has exited, hotkeys are dead");
    });

    Ok(())
}

/// Apply one hotkey intent to the runtime. Serial by construction: the caller
/// awaits each action before reading the next.
async fn dispatch_hotkey(rt: &runtime::AppRuntime, action: hotkey::HotkeyAction) {
    match action {
        hotkey::HotkeyAction::Toggle => {
            tracing::info!(action = "toggle", "hotkey pressed");
            let _ = rt.toggle_dictation().await;
        }
        hotkey::HotkeyAction::Start => {
            tracing::info!(action = "start", "hotkey pressed");
            // The runtime surfaces this to the UI itself; log it here so the
            // unsanitized cause is in the trace too.
            if let Err(e) = rt.start_dictation().await {
                tracing::warn!(error = %e, "hotkey start failed");
            }
        }
        hotkey::HotkeyAction::Stop => {
            tracing::info!(action = "stop", "hotkey pressed");
            let _ = rt.stop_and_wait_dictation().await;
        }
        // Harmless when idle, but logging every idle Escape is noise.
        hotkey::HotkeyAction::Cancel if rt.is_active().await => {
            tracing::info!(action = "cancel", "hotkey pressed");
            let _ = rt.cancel_dictation().await;
        }
        hotkey::HotkeyAction::Cancel => {}
    }
}

/// The global-shortcut plugin as the hotkey controller sees it.
struct PluginShortcuts {
    app: tauri::AppHandle,
    /// The last pair that registered successfully. `register_hotkeys_with`
    /// unbinds before registering, so a combo another application already
    /// owns would otherwise leave no hotkeys at all.
    bound: std::sync::Mutex<Option<lune_core::config::HotkeyConfig>>,
}

impl hotkey::PluginHotkeys for PluginShortcuts {
    fn rebind(&self, config: &lune_core::config::HotkeyConfig) -> Result<(), String> {
        let mut bound = self.bound.lock().unwrap_or_else(|e| e.into_inner());
        match register_hotkeys_with(&self.app, config) {
            Ok(()) => {
                *bound = Some(config.clone());
                Ok(())
            }
            Err(e) => {
                if let Some(previous) = bound.as_ref()
                    && let Err(restore) = register_hotkeys_with(&self.app, previous)
                {
                    tracing::warn!(error = %restore, "could not restore the previous hotkeys");
                    *bound = None;
                }
                Err(e.to_string())
            }
        }
    }

    fn clear(&self) {
        *self.bound.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if let Err(e) = self.app.global_shortcut().unregister_all() {
            tracing::warn!(error = %e, "could not release the global-shortcut hotkeys");
        }
    }
}

/// Toggle-only fallback for platforms with no native backend.
///
/// The plugin reports a single "activated" callback with no matching release,
/// so `ActivationMode::Hold` cannot be honoured here and is not attempted.
///
/// Both combos are parsed before anything is unbound, so a rejected binding
/// leaves the previous pair working rather than nothing at all.
fn register_hotkeys_with(
    app: &tauri::AppHandle,
    config: &lune_core::config::HotkeyConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let toggle_str = format_hotkey(&config.toggle_dictation);
    let cancel_str = format_hotkey(&config.cancel_dictation);

    let toggle: Shortcut = toggle_str
        .parse()
        .map_err(|e| format!("parse toggle hotkey: {e}"))?;
    let cancel: Shortcut = cancel_str
        .parse()
        .map_err(|e| format!("parse cancel hotkey: {e}"))?;

    // The plugin rejects a shortcut that is already registered, so a rebind has
    // to drop the previous pair first. Nothing else in this app registers one.
    app.global_shortcut().unregister_all()?;

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
