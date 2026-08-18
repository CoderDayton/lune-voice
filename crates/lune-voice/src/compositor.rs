//! Whether the app must fall back to XWayland for the dictation pill.
//!
//! The pill overlay needs two things a *native* Wayland window cannot do on its
//! own: place itself at an absolute position, and stay always-on-top. Both are
//! only reachable through the `wlr-layer-shell` protocol. wlroots-based
//! compositors and COSMIC implement it; GNOME/Mutter and most KDE builds do
//! not. Where it is missing we run the whole app under XWayland
//! (`GDK_BACKEND=x11`) so ordinary X11 window positioning and
//! `_NET_WM_STATE_ABOVE` (always-on-top) work again.
//!
//! The probe is a one-shot Wayland registry query — no GTK, so it can run
//! before GDK is initialised (which is exactly when the backend must be chosen).

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    globals::{GlobalListContents, registry_queue_init},
    protocol::wl_registry::WlRegistry,
};

/// Wire name of the layer-shell global (equals `ZwlrLayerShellV1::interface()`'s
/// name). The protocol name is part of the wire ABI, so this literal is stable.
const LAYER_SHELL_INTERFACE: &str = "zwlr_layer_shell_v1";

/// True only when this is a Wayland session whose compositor does **not**
/// advertise `wlr-layer-shell` — the case where the pill can't be a native
/// Wayland overlay and the app should run under XWayland instead.
///
/// False on X11 (no `WAYLAND_DISPLAY`), on compositors that implement
/// layer-shell (wlroots/COSMIC), and — conservatively — whenever the compositor
/// can't be probed, so a transient Wayland error never forces X11 onto a
/// compositor that actually supports the native path. Also false when the X
/// display refuses the connection: an unreachable XWayland costs the pill its
/// placement, but forcing GDK onto it costs the user the whole app.
pub fn should_use_xwayland_fallback() -> bool {
    xwayland_fallback_decision(
        std::env::var_os("WAYLAND_DISPLAY").is_some(),
        layer_shell_advertised(),
        x11_display_usable,
    )
}

/// The rule itself, separated from the probes so it can be exercised without a
/// compositor or an X server.
fn xwayland_fallback_decision(
    wayland_session: bool,
    layer_shell: Option<bool>,
    x11_reachable: impl FnOnce() -> bool,
) -> bool {
    if !wayland_session {
        return false;
    }
    if !matches!(layer_shell, Some(false)) {
        return false;
    }
    if !x11_reachable() {
        tracing::warn!(
            "Wayland without layer-shell, but the X display cannot be opened — \
             staying on native Wayland. The dictation pill cannot position \
             itself or stay on top here; check DISPLAY and XAUTHORITY"
        );
        return false;
    }
    true
}

/// `Some(true)`/`Some(false)` once we positively know layer-shell is
/// present/absent; `None` if the Wayland compositor couldn't be queried.
fn layer_shell_advertised() -> Option<bool> {
    let conn = Connection::connect_to_env().ok()?;
    // `registry_queue_init` performs the initial roundtrip, so the returned
    // global list already reflects everything the compositor advertises.
    let (globals, _queue) = registry_queue_init::<Probe>(&conn).ok()?;
    Some(
        globals
            .contents()
            .with_list(|list| list.iter().any(|g| g.interface == LAYER_SHELL_INTERFACE)),
    )
}

/// The probe holds no state — `registry_queue_init` collects the global list
/// itself; we only need a type that implements the registry `Dispatch`.
struct Probe;

impl Dispatch<WlRegistry, GlobalListContents> for Probe {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: <WlRegistry as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

/// Whether an X11 display can actually be opened, authorization included.
///
/// A socket that merely accepts the connection is not enough: an Xwayland
/// server whose cookie file has been rotated out from under a long-lived
/// environment accepts the client and then refuses it, and the refusal only
/// surfaces at `gtk::init` — which `tao` unwraps, so the whole app dies before
/// its first window. `x11rb` performs the same setup exchange GDK would,
/// reading `DISPLAY` and `XAUTHORITY` the same way, so it fails exactly where
/// GDK fails.
fn x11_display_usable() -> bool {
    probe_with_timeout(X11_PROBE_TIMEOUT, || {
        x11rb::rust_connection::RustConnection::connect(None).is_ok()
    })
}

/// How long the X probe waits before giving up on an answer.
///
/// A live X server completes the setup exchange over a local socket in
/// microseconds, so this is only ever spent on one that has stopped answering.
const X11_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Run `probe` on a thread we are willing to abandon, answering `false` if it
/// does not report back in time.
///
/// `RustConnection::connect` has no deadline of its own, and neither does the
/// X handshake: a server that accepts the socket and then stalls — a wedged
/// XWayland, a display forwarded over a dead SSH connection — would block
/// startup on the main thread before any window exists. That is the same class
/// of failure the probe was added to prevent, just silent instead of loud, so
/// a slow answer is treated as no answer and the app stays on Wayland.
fn probe_with_timeout(timeout: Duration, probe: impl FnOnce() -> bool + Send + 'static) -> bool {
    let (tx, rx) = mpsc::channel();
    // The receiver going away is the timeout case; the send failing there is
    // expected, not an error.
    std::thread::spawn(move || {
        let _ = tx.send(probe());
    });
    match rx.recv_timeout(timeout) {
        Ok(usable) => usable,
        Err(RecvTimeoutError::Timeout) => {
            tracing::warn!(
                timeout_secs = timeout.as_secs(),
                "the X display did not answer in time; treating it as unreachable"
            );
            false
        }
        // The sender was dropped without a value, so the probe panicked.
        // Reporting it as a timeout would send debugging after a hang that
        // never happened.
        Err(RecvTimeoutError::Disconnected) => {
            tracing::warn!("the X display probe panicked; treating the display as unreachable");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{probe_with_timeout, xwayland_fallback_decision};
    use std::time::Duration;

    #[test]
    fn x11_session_keeps_its_native_backend() {
        assert!(!xwayland_fallback_decision(false, None, || true));
        assert!(!xwayland_fallback_decision(false, Some(false), || true));
    }

    #[test]
    fn layer_shell_compositor_stays_on_wayland() {
        assert!(!xwayland_fallback_decision(true, Some(true), || true));
    }

    #[test]
    fn unprobeable_compositor_stays_on_wayland() {
        assert!(!xwayland_fallback_decision(true, None, || true));
    }

    #[test]
    fn missing_layer_shell_falls_back_when_x11_answers() {
        assert!(xwayland_fallback_decision(true, Some(false), || true));
    }

    #[test]
    fn unreachable_x11_keeps_wayland_instead_of_panicking() {
        // Forcing GDK_BACKEND=x11 at an X display that refuses the client
        // aborts the app at gtk::init. A pill that can't position itself is
        // worth far more than that.
        assert!(!xwayland_fallback_decision(true, Some(false), || false));
    }

    #[test]
    fn a_probe_that_answers_is_believed() {
        assert!(probe_with_timeout(Duration::from_secs(30), || true));
        assert!(!probe_with_timeout(Duration::from_secs(30), || false));
    }

    #[test]
    fn a_stalled_probe_gives_up_instead_of_blocking_startup() {
        let started = std::time::Instant::now();
        let answered = probe_with_timeout(Duration::from_millis(50), || {
            std::thread::sleep(Duration::from_secs(30));
            true
        });
        assert!(
            !answered,
            "a probe that never answers must not report success"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the probe blocked for {:?} instead of timing out",
            started.elapsed()
        );
    }
}
