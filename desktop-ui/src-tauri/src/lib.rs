mod commands;
mod dto;
mod feedback;
#[cfg(target_os = "linux")]
mod hotkey_linux;
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
            commands::initialize_pipeline,
            commands::gpu_backend_status,
            commands::uinput_access_ready,
            commands::grant_uinput_access,
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
