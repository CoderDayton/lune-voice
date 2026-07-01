// Prevents additional console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            // Quiet by default: warnings + errors only, for the app and for
            // whisper.cpp/ggml (which route into tracing). Override with RUST_LOG
            // for diagnostics, e.g. `RUST_LOG=info` or `RUST_LOG=lune_app=debug`.
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
        )
        .init();
    lune_ui::run();
}
