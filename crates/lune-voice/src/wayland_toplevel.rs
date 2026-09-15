// SPDX-License-Identifier: GPL-3.0-or-later

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

use std::time::Instant;

use tracing::debug;

use crate::toplevel::{QUERY_BUDGET, Toplevels};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, globals::registry_queue_init};
use wayland_protocols_wlr::foreign_toplevel::v1::client::{
    zwlr_foreign_toplevel_handle_v1::{self, ZwlrForeignToplevelHandleV1},
    zwlr_foreign_toplevel_manager_v1::{self, ZwlrForeignToplevelManagerV1},
};

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
    crate::toplevel::active_app_id("the wlr toplevel manager", query)
}

fn query() -> Option<String> {
    let started = Instant::now();
    let conn = Connection::connect_to_env().ok()?;
    let (globals, mut queue) = registry_queue_init::<Toplevels>(&conn).ok()?;
    let qh = queue.handle();

    let manager: ZwlrForeignToplevelManagerV1 =
        match globals.bind(&qh, 1..=MAX_PROTOCOL_VERSION, ()) {
            Ok(m) => m,
            Err(e) => {
                debug!(error = %e, "wlr-foreign-toplevel-management not advertised");
                return None;
            }
        };

    let mut state = Toplevels::default();

    // Roundtrip 1: compositor fires `toplevel` for each existing toplevel.
    queue.roundtrip(&mut state).ok()?;

    // Subsequent roundtrips: each handle fires `app_id` / `state` / `done`.
    // We finish when every known toplevel has produced a `done` event, or
    // the time budget elapses.
    while !state.all_done() {
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

    state.activated_app_id()
}

// ─── Dispatch: manager ───────────────────────────────────────────────────────

impl Dispatch<ZwlrForeignToplevelManagerV1, ()> for Toplevels {
    fn event(
        s: &mut Self,
        _: &ZwlrForeignToplevelManagerV1,
        event: zwlr_foreign_toplevel_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_foreign_toplevel_manager_v1::Event::Toplevel { toplevel } = event {
            s.track(toplevel.id().protocol_id());
        }
        // Event::Finished is irrelevant for a one-shot query.
    }
}

// ─── Dispatch: per-toplevel handle ───────────────────────────────────────────

impl Dispatch<ZwlrForeignToplevelHandleV1, ()> for Toplevels {
    fn event(
        s: &mut Self,
        handle: &ZwlrForeignToplevelHandleV1,
        event: zwlr_foreign_toplevel_handle_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let id = handle.id().protocol_id();
        match event {
            zwlr_foreign_toplevel_handle_v1::Event::AppId { app_id } => s.set_app_id(id, app_id),
            zwlr_foreign_toplevel_handle_v1::Event::State { state } => s.set_state(id, &state),
            zwlr_foreign_toplevel_handle_v1::Event::Done => s.set_done(id),
            zwlr_foreign_toplevel_handle_v1::Event::Closed => s.forget(id),
            // Title / OutputEnter / OutputLeave / Parent: ignored.
            _ => {}
        }
    }
}
