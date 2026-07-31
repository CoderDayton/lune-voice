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
/// compositor that actually supports the native path.
pub fn should_use_xwayland_fallback() -> bool {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        return false;
    }
    matches!(layer_shell_advertised(), Some(false))
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
