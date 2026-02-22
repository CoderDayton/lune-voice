use lune_core::config::DictationMode;
use tokio::process::Command;
use tracing::warn;

/// Deliver finalized text according to the active dictation mode.
pub async fn deliver(text: &str, mode: DictationMode) {
    match mode {
        DictationMode::TypeInPlace => type_with_wtype(text).await,
        DictationMode::Clipboard => copy_with_wl_copy(text).await,
        DictationMode::Both => {
            tokio::join!(type_with_wtype(text), copy_with_wl_copy(text));
        }
        _ => {
            warn!("unknown dictation mode {mode:?}, falling back to TypeInPlace");
            type_with_wtype(text).await;
        }
    }
}

/// Type text into the focused input via `wtype -`.
///
/// Reads from stdin to safely handle multi-line text and special characters.
async fn type_with_wtype(text: &str) {
    let result = Command::new("wtype")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match result {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "failed to spawn wtype — is it installed?");
            return;
        }
    };

    // Write text to stdin then close.
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        if let Err(e) = stdin.write_all(text.as_bytes()).await {
            warn!(error = %e, "failed to write to wtype stdin");
            return;
        }
        drop(stdin);
    }

    match child.wait().await {
        Ok(status) if !status.success() => {
            warn!(?status, "wtype exited with non-zero status");
        }
        Err(e) => {
            warn!(error = %e, "failed to wait on wtype");
        }
        _ => {}
    }
}

/// Copy text to the Wayland clipboard via `wl-copy -- <text>`.
///
/// The `--` prevents text starting with `-` from being interpreted as flags.
async fn copy_with_wl_copy(text: &str) {
    let result = Command::new("wl-copy")
        .arg("--")
        .arg(text)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .status()
        .await;

    match result {
        Ok(status) if !status.success() => {
            warn!(?status, "wl-copy exited with non-zero status");
        }
        Err(e) => {
            warn!(error = %e, "failed to run wl-copy — is it installed?");
        }
        _ => {}
    }
}
