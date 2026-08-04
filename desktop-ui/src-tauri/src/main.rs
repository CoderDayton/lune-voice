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

fn main() {
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
