// SPDX-License-Identifier: GPL-3.0-or-later

// Prevents additional console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use tracing_subscriber::fmt::writer::MakeWriterExt;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Rotate the log once it passes this size. One generation is kept, which
/// covers "it broke, then I restarted it" without growing without bound.
const MAX_LOG_BYTES: u64 = 4 * 1024 * 1024;

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

/// Open the on-disk log, rotating the previous one aside when it has grown
/// past [`MAX_LOG_BYTES`].
///
/// Launched from a desktop entry, this process has no console: stderr goes
/// nowhere the user can reach, so "dictation stopped working" would arrive
/// with nothing to look at. Returns `None` when no data directory resolves or
/// the file cannot be opened — logging then continues on stderr alone rather
/// than taking the app down.
fn open_log_file() -> Option<PathBuf> {
    let dir = lune_persist::AppPaths::resolve()?.data_dir().join("logs");
    fs::create_dir_all(&dir).ok()?;
    let path = dir.join("lune-voice.log");
    if fs::metadata(&path).is_ok_and(|m| m.len() > MAX_LOG_BYTES) {
        // Best-effort: a failed rotation just means the file keeps growing.
        let _ = fs::rename(&path, path.with_extension("log.1"));
    }
    Some(path)
}

/// Route panics through tracing before the default hook runs.
///
/// Release builds unwind, so a panicking session is caught by its supervisor
/// and the app keeps running — which is exactly why the panic needs recording:
/// without this it leaves no location, no backtrace, and no indication it
/// happened at all beyond one dictation quietly failing. A panic that no
/// supervisor catches still ends the process, and this is the only thing that
/// says why.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map_or_else(|| "unknown".to_owned(), |l| l.to_string());
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!(location, %backtrace, "panic: {info}");
        default_hook(info);
    }));
}

fn main() {
    // No WebKitGTK rendering overrides. `WEBKIT_DISABLE_DMABUF_RENDERER` +
    // `__NV_DISABLE_EXPLICIT_SYNC` used to be set here to coax an alpha channel
    // out of the pill overlay's webview on the Nvidia driver; they cost the
    // whole app its GPU compositing and still left the pill repainting in stale
    // bands. The pill is drawn natively with cairo on Linux now
    // (`pill_native.rs`), and the main window is fully opaque and never needed
    // either flag.
    harden_file_permissions();

    let log_file = open_log_file().and_then(|path| {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
    });
    let builder = tracing_subscriber::fmt()
        .with_env_filter(
            // Quiet by default: warnings + errors only, for the app and for
            // whisper.cpp/ggml (which route into tracing). Override with RUST_LOG
            // for diagnostics, e.g. `RUST_LOG=info` or `RUST_LOG=lune_voice=debug`.
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
        )
        // Off so the log file holds plain text rather than escape sequences;
        // one writer feeds both sinks.
        .with_ansi(false);
    match log_file {
        Some(file) => builder
            .with_writer(std::io::stderr.and(Arc::new(file)))
            .init(),
        None => builder.init(),
    }

    install_panic_hook();
    lune_ui::run();
}
