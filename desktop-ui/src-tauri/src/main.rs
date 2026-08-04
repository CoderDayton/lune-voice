// Prevents additional console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Restrict the process umask so every file this app creates — the transcripts
/// DB, its WAL/SHM sidecars, settings, model downloads — is owner-only by
/// default, regardless of the user's shell umask. The DB is also chmodded
/// explicitly in `Database::open`; this covers the WAL/SHM sidecars SQLite
/// creates lazily during writes, which would otherwise inherit the shell umask.
#[cfg(unix)]
fn harden_file_permissions() {
    // SAFETY: `umask` is a plain libc call with no memory-unsafety; passing a
    // constant mode is well-defined, and the previous mask is discarded.
    unsafe { libc::umask(0o077) };
}

#[cfg(not(unix))]
fn harden_file_permissions() {}

/// WebKitGTK variables that restore correct alpha compositing for the pill
/// overlay.
///
/// On the Nvidia proprietary driver the accelerated compositing path drops the
/// alpha channel: a transparent window paints correctly for the first
/// (software) frame and is then replaced by an opaque surface, and translucent
/// fills ghost or pick up a colour cast from a channel-order mismatch. Measured
/// on WebKitGTK 2.52 / Wayland / driver 595: neither variable alone is enough —
/// `COMPOSITING_MODE` leaves interactive fills painting opaque, and
/// `DMABUF_RENDERER` alone does not restore window alpha. Together the overlay
/// composites correctly.
///
/// The cost is that CSS animation and compositing run on the CPU app-wide. That
/// is negligible for the pill; if the main window's animations ever suffer, the
/// alternative is giving the pill its own process with these set only there.
///
/// An explicit value from the environment always wins, so this can be overridden
/// (or re-enabled) from the launcher without touching the binary.
#[cfg(target_os = "linux")]
fn configure_webkit_rendering() {
    const WEBKIT_RENDERING_VARS: [(&str, &str); 2] = [
        ("WEBKIT_DISABLE_COMPOSITING_MODE", "1"),
        ("WEBKIT_DISABLE_DMABUF_RENDERER", "1"),
    ];
    for (key, value) in WEBKIT_RENDERING_VARS {
        if std::env::var_os(key).is_none() {
            // SAFETY: `set_var` is unsound only with concurrent env access. This
            // is the first statement in `main`, before the tokio runtime, GTK,
            // or the webview exist, so the process is still single-threaded.
            unsafe { std::env::set_var(key, value) };
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn configure_webkit_rendering() {}

fn main() {
    // Must precede every other initialisation: WebKitGTK reads these when GTK
    // starts up, so setting them later has no effect.
    configure_webkit_rendering();
    harden_file_permissions();
    tracing_subscriber::fmt()
        .with_env_filter(
            // Quiet by default: warnings + errors only, for the app and for
            // whisper.cpp/ggml (which route into tracing). Override with RUST_LOG
            // for diagnostics, e.g. `RUST_LOG=info` or `RUST_LOG=lune_voice=debug`.
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
        )
        .init();
    lune_ui::run();
}
