//! COSMIC `zcosmic_toplevel_info_v1` client.
//!
//! cosmic-comp does not implement `zwlr_foreign_toplevel_manager_v1`. It
//! exposes its own `zcosmic_toplevel_info_v1` (and the standardized
//! `ext_foreign_toplevel_list_v1` umbrella). This module queries the COSMIC
//! protocol directly to recover the activated toplevel's `app_id`.
//!
//! We bind the protocol at **version 1** intentionally: in v1 the manager
//! emits `Toplevel { handle: zcosmic_toplevel_handle_v1 }` events directly,
//! and each handle ships `app_id` / `state` events. Higher versions
//! deprecate this in favour of bridging through `ext_foreign_toplevel_list`,
//! which would double the protocol surface (two managers, two handle
//! lifetimes, plus `get_cosmic_toplevel` calls) for zero functional gain on
//! a one-shot read. cosmic-comp keeps v1 fully functional for backward
//! compatibility, so this is the lean option.
//!
//! Like the wlr client, this opens a fresh connection per call and bounds
//! its work by a hard time budget — see `wayland_toplevel.rs` for the
//! "why per-call" discussion.

use std::time::Instant;

use crate::cosmic_protocol::toplevel_info::client::{
    zcosmic_toplevel_handle_v1::{self, ZcosmicToplevelHandleV1},
    zcosmic_toplevel_info_v1::{self, ZcosmicToplevelInfoV1},
};
use tracing::debug;

use crate::toplevel::{QUERY_BUDGET, Toplevels};
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle, event_created_child, globals::registry_queue_init,
};

/// Returns the `app_id` of the activated toplevel via cosmic-toplevel-info,
/// or `None` if the protocol is not advertised, no toplevel is activated,
/// the session is not Wayland, or any wayland error occurs.
pub fn cosmic_active_app_id() -> Option<String> {
    crate::toplevel::active_app_id("the cosmic toplevel manager", query)
}

fn query() -> Option<String> {
    let started = Instant::now();
    let conn = Connection::connect_to_env().ok()?;
    let (globals, mut queue) = registry_queue_init::<Toplevels>(&conn).ok()?;
    let qh = queue.handle();

    let info: ZcosmicToplevelInfoV1 = match globals.bind(&qh, 1..=1, ()) {
        Ok(i) => i,
        Err(e) => {
            debug!(error = %e, "zcosmic_toplevel_info_v1 not advertised");
            return None;
        }
    };

    let mut state = Toplevels::default();

    // Roundtrip until we observe an activated toplevel with an app_id, or
    // exhaust the budget. Each roundtrip drains a layer of the protocol:
    // 1. globals already settled by registry_queue_init
    // 2. manager emits Toplevel events (one per existing toplevel)
    // 3+ each handle emits app_id / title / state / output_enter / done
    for _ in 0..6 {
        if started.elapsed() >= QUERY_BUDGET {
            debug!("cosmic-toplevel query budget exhausted");
            break;
        }
        queue.roundtrip(&mut state).ok()?;
        if state.found_activated() {
            // One extra roundtrip to absorb any final state events that
            // might overwrite the activated flag (e.g., minimized=false
            // arriving after activated=true on the same window).
            let _ = queue.roundtrip(&mut state);
            break;
        }
    }

    info.stop();
    let _ = queue.flush();

    state.activated_app_id()
}

// ─── Dispatch: manager ───────────────────────────────────────────────────────

impl Dispatch<ZcosmicToplevelInfoV1, ()> for Toplevels {
    fn event(
        s: &mut Self,
        _: &ZcosmicToplevelInfoV1,
        event: zcosmic_toplevel_info_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zcosmic_toplevel_info_v1::Event::Toplevel { toplevel } = event {
            s.track(toplevel.id().protocol_id());
        }
    }

    event_created_child!(
        Toplevels,
        ZcosmicToplevelInfoV1,
        [
            zcosmic_toplevel_info_v1::EVT_TOPLEVEL_OPCODE => (ZcosmicToplevelHandleV1, ()),
        ]
    );
}

// ─── Dispatch: per-toplevel handle ───────────────────────────────────────────

impl Dispatch<ZcosmicToplevelHandleV1, ()> for Toplevels {
    fn event(
        s: &mut Self,
        handle: &ZcosmicToplevelHandleV1,
        event: zcosmic_toplevel_handle_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let id = handle.id().protocol_id();
        match event {
            zcosmic_toplevel_handle_v1::Event::AppId { app_id } => s.set_app_id(id, app_id),
            zcosmic_toplevel_handle_v1::Event::State { state } => s.set_state(id, &state),
            zcosmic_toplevel_handle_v1::Event::Closed => s.forget(id),
            // Title / OutputEnter / OutputLeave / WorkspaceEnter /
            // WorkspaceLeave / Done / Geometry: ignored.
            _ => {}
        }
    }
}
