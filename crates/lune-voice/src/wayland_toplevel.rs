//! Wayland `wlr-foreign-toplevel-management-v1` client.
//!
//! Returns the `app_id` of the focused (activated) toplevel on any compositor
//! that implements the protocol — this includes Hyprland, Sway, COSMIC, KWin
//! (KDE 5.27+), Wayfire, river, and Mir-based compositors.
//!
//! Used as the primary detection source on COSMIC, where no IPC/CLI tool
//! exposes focused-window info, and as a generic Wayland fallback before the
//! X11 (xdotool) probe.
//!
//! ## Why a fresh connection per call
//! `focused_window_class()` is invoked once per dictation event (hotkey press
//! or delivery time). Opening a Wayland socket is local IPC and typically
//! settles in under 10 ms even with a few dozen toplevels. Keeping a
//! long-lived connection would require an extra background thread plus a
//! shared state mutex; not worth the complexity until profiling shows
//! per-call cost is hot.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::{debug, trace};

use crate::probe;
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    globals::{GlobalListContents, registry_queue_init},
    protocol::wl_registry::WlRegistry,
};
use wayland_protocols_wlr::foreign_toplevel::v1::client::{
    zwlr_foreign_toplevel_handle_v1::{self, ZwlrForeignToplevelHandleV1},
    zwlr_foreign_toplevel_manager_v1::{self, ZwlrForeignToplevelManagerV1},
};

/// `state` array enum value for an activated (focused) toplevel.
const STATE_ACTIVATED: u32 = 2;

/// Soft budget, checked between roundtrips. Compositor IPC should finish in
/// single digit ms; anything beyond this means something is wrong and we'd
/// rather fall through than block delivery.
const QUERY_BUDGET: Duration = Duration::from_millis(150);

/// Hard ceiling, enforced from outside the query.
///
/// `roundtrip` blocks on the socket with no deadline, so a compositor that
/// stops answering *mid*-roundtrip never reaches the budget check above. Only
/// a bound the query cannot influence keeps a wedged compositor from freezing
/// delivery. Set above the soft budget so a healthy-but-slow query still gets
/// to finish on its own terms.
const QUERY_DEADLINE: Duration = Duration::from_millis(500);

/// Maximum protocol version we negotiate with. Bumping is safe — newer
/// versions stay backward compatible at the wire level.
const MAX_PROTOCOL_VERSION: u32 = 3;

/// Returns the `app_id` of the activated toplevel, or `None` if the
/// compositor does not implement the protocol, no toplevel is activated,
/// or any wayland error occurs.
///
/// Safe to call on non-Wayland sessions: returns `None` immediately when
/// `WAYLAND_DISPLAY` is unset.
pub fn wlr_active_app_id() -> Option<String> {
    std::env::var_os("WAYLAND_DISPLAY")?;

    let started = Instant::now();
    let result = probe::bounded(QUERY_DEADLINE, "the wlr toplevel manager", None, query);
    trace!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        ?result,
        "wlr query done"
    );
    result
}

fn query() -> Option<String> {
    let started = Instant::now();
    let conn = Connection::connect_to_env().ok()?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn).ok()?;
    let qh = queue.handle();

    let manager: ZwlrForeignToplevelManagerV1 =
        match globals.bind(&qh, 1..=MAX_PROTOCOL_VERSION, ()) {
            Ok(m) => m,
            Err(e) => {
                debug!(error = %e, "wlr-foreign-toplevel-management not advertised");
                return None;
            }
        };

    let mut state = State::default();

    // Roundtrip 1: compositor fires `toplevel` for each existing toplevel.
    queue.roundtrip(&mut state).ok()?;

    // Subsequent roundtrips: each handle fires `app_id` / `state` / `done`.
    // We finish when every known toplevel has produced a `done` event, or
    // the time budget elapses.
    while !state.toplevels.values().all(|t| t.done) {
        if started.elapsed() >= QUERY_BUDGET {
            debug!("wlr query budget exhausted before all `done` events arrived");
            break;
        }
        queue.roundtrip(&mut state).ok()?;
    }

    // Tell the compositor to stop streaming events; the proxy is destroyed
    // automatically when `manager` drops.
    manager.stop();
    let _ = queue.flush();

    state
        .toplevels
        .into_values()
        .find(|t| t.activated)
        .and_then(|t| t.app_id)
}

#[derive(Default)]
struct ToplevelInfo {
    app_id: Option<String>,
    activated: bool,
    done: bool,
}

#[derive(Default)]
struct State {
    toplevels: HashMap<u32, ToplevelInfo>,
}

// ─── Dispatch: registry (no-op; registry_queue_init handles bookkeeping) ─────

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: <WlRegistry as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // GlobalListContents tracks adds/removes internally. We do not care
        // about hot-plugged globals during a one-shot query.
    }
}

// ─── Dispatch: manager ───────────────────────────────────────────────────────

impl Dispatch<ZwlrForeignToplevelManagerV1, ()> for State {
    fn event(
        s: &mut Self,
        _: &ZwlrForeignToplevelManagerV1,
        event: zwlr_foreign_toplevel_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_foreign_toplevel_manager_v1::Event::Toplevel { toplevel } = event {
            s.toplevels
                .insert(toplevel.id().protocol_id(), ToplevelInfo::default());
        }
        // Event::Finished is irrelevant for a one-shot query.
    }
}

// ─── Dispatch: per-toplevel handle ───────────────────────────────────────────

impl Dispatch<ZwlrForeignToplevelHandleV1, ()> for State {
    fn event(
        s: &mut Self,
        handle: &ZwlrForeignToplevelHandleV1,
        event: zwlr_foreign_toplevel_handle_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let id = handle.id().protocol_id();
        let entry = s.toplevels.entry(id).or_default();
        match event {
            zwlr_foreign_toplevel_handle_v1::Event::AppId { app_id } => {
                entry.app_id = Some(app_id);
            }
            zwlr_foreign_toplevel_handle_v1::Event::State { state } => {
                // The protocol delivers the state array as raw bytes
                // (`wl_array`). Each entry is a u32 in native byte order.
                entry.activated = state
                    .chunks_exact(4)
                    .any(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]) == STATE_ACTIVATED);
            }
            zwlr_foreign_toplevel_handle_v1::Event::Done => {
                entry.done = true;
            }
            zwlr_foreign_toplevel_handle_v1::Event::Closed => {
                s.toplevels.remove(&id);
            }
            // Title / OutputEnter / OutputLeave / Parent: ignored.
            _ => {}
        }
    }
}
