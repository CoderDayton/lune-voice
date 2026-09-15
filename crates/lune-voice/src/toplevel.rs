// SPDX-License-Identifier: GPL-3.0-or-later

//! Shared state for the two foreign-toplevel clients.
//!
//! wlr-foreign-toplevel-management and COSMIC's toplevel-info answer the same
//! question — which window is activated, and what is its `app_id` — through two
//! unrelated sets of protocol objects. Everything either side of those objects
//! is the same: the per-toplevel record, the activated-state parse, the time
//! budget, and the registry dispatch. Both clients drive this one state type,
//! so a fix to the shared half lands in both.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::trace;
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle, globals::GlobalListContents,
    protocol::wl_registry::WlRegistry,
};

use crate::probe;

/// `state` array enum value for an activated (focused) toplevel. Both
/// protocols reuse xdg-toplevel's numbering.
const STATE_ACTIVATED: u32 = 2;

/// Soft budget, checked between roundtrips. Compositor IPC should finish in
/// single-digit ms; anything beyond this means something is wrong and we'd
/// rather fall through than block delivery.
pub(crate) const QUERY_BUDGET: Duration = Duration::from_millis(150);

/// Hard ceiling, enforced from outside the query.
///
/// `roundtrip` blocks on the socket with no deadline, so a compositor that
/// stops answering *mid*-roundtrip never reaches the budget check. Only a
/// bound the query cannot influence keeps a wedged compositor from freezing
/// delivery. Set above the soft budget so a healthy-but-slow query still gets
/// to finish on its own terms.
const QUERY_DEADLINE: Duration = Duration::from_millis(500);

/// What one toplevel has told us so far.
#[derive(Default)]
struct ToplevelInfo {
    app_id: Option<String>,
    activated: bool,
    /// Set by the wlr `done` event, which marks the end of that toplevel's
    /// event burst. The COSMIC client does not wait on it.
    done: bool,
}

/// Every toplevel the compositor has announced, keyed by protocol id.
#[derive(Default)]
pub(crate) struct Toplevels {
    windows: HashMap<u32, ToplevelInfo>,
}

impl Toplevels {
    /// Start tracking a toplevel the manager just announced.
    pub(crate) fn track(&mut self, id: u32) {
        self.windows.insert(id, ToplevelInfo::default());
    }

    pub(crate) fn set_app_id(&mut self, id: u32, app_id: String) {
        self.windows.entry(id).or_default().app_id = Some(app_id);
    }

    /// Record whether a toplevel `state` array marks the window activated.
    ///
    /// The protocol delivers the array as raw bytes (`wl_array`); each entry is
    /// a u32 in native byte order.
    pub(crate) fn set_state(&mut self, id: u32, state: &[u8]) {
        self.windows.entry(id).or_default().activated = state
            .chunks_exact(4)
            .any(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]) == STATE_ACTIVATED);
    }

    pub(crate) fn set_done(&mut self, id: u32) {
        self.windows.entry(id).or_default().done = true;
    }

    pub(crate) fn forget(&mut self, id: u32) {
        self.windows.remove(&id);
    }

    /// Whether every known toplevel has finished its event burst.
    pub(crate) fn all_done(&self) -> bool {
        self.windows.values().all(|t| t.done)
    }

    /// Whether the answer is already in hand.
    pub(crate) fn found_activated(&self) -> bool {
        self.windows
            .values()
            .any(|t| t.activated && t.app_id.is_some())
    }

    /// The activated toplevel's `app_id`, once the query is over.
    ///
    /// More than one toplevel can carry the activated flag — a compositor with
    /// several outputs marks one per output, and a stale flag survives until
    /// the deactivating update arrives. An activated window the compositor
    /// never named answers nothing, so it is skipped rather than accepted as
    /// the answer; otherwise `HashMap` order alone would decide whether the
    /// query returns a class or `None`.
    pub(crate) fn activated_app_id(self) -> Option<String> {
        self.windows
            .into_values()
            .filter(|t| t.activated)
            .find_map(|t| t.app_id)
    }
}

/// Run one toplevel query under the hard deadline, on a Wayland session only.
///
/// `subject` names the manager in the give-up log and the timing trace.
pub(crate) fn active_app_id(
    subject: &'static str,
    query: impl FnOnce() -> Option<String> + Send + 'static,
) -> Option<String> {
    std::env::var_os("WAYLAND_DISPLAY")?;

    let started = Instant::now();
    let result = probe::bounded(QUERY_DEADLINE, subject, None, query);
    trace!(
        subject,
        elapsed_ms = started.elapsed().as_millis() as u64,
        ?result,
        "toplevel query done"
    );
    result
}

// `registry_queue_init` tracks adds/removes inside `GlobalListContents`, and a
// one-shot query does not care about globals hot-plugged while it runs.
impl Dispatch<WlRegistry, GlobalListContents> for Toplevels {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn wl_array(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    #[test]
    fn the_activated_flag_is_read_out_of_the_state_array() {
        let mut t = Toplevels::default();
        t.track(1);
        t.set_state(1, &wl_array(&[4, STATE_ACTIVATED, 1]));
        assert!(t.windows[&1].activated);

        t.set_state(1, &wl_array(&[4, 1]));
        assert!(!t.windows[&1].activated, "a later state must clear it");
    }

    /// `wl_array` carries a byte count, not a u32 count. A trailing partial
    /// entry must not be read as a state.
    #[test]
    fn a_ragged_state_array_reads_only_whole_entries() {
        let mut t = Toplevels::default();
        let mut bytes = wl_array(&[1]);
        bytes.push(STATE_ACTIVATED as u8);
        t.set_state(1, &bytes);
        assert!(!t.windows[&1].activated);
        t.set_state(2, &[]);
        assert!(!t.windows[&2].activated);
    }

    #[test]
    fn the_activated_toplevel_with_an_app_id_is_the_answer() {
        let mut t = Toplevels::default();
        t.track(1);
        t.set_app_id(1, "firefox".into());
        t.track(2);
        t.set_app_id(2, "kitty".into());
        t.set_state(2, &wl_array(&[STATE_ACTIVATED]));
        assert!(t.found_activated());
        assert_eq!(t.activated_app_id().as_deref(), Some("kitty"));
    }

    /// An activated window the compositor never named cannot answer the
    /// question, so the query must keep waiting rather than settle for it.
    #[test]
    fn an_activated_toplevel_without_an_app_id_is_not_yet_an_answer() {
        let mut t = Toplevels::default();
        t.track(1);
        t.set_state(1, &wl_array(&[STATE_ACTIVATED]));
        assert!(!t.found_activated());
        assert_eq!(t.activated_app_id(), None);
    }

    /// Several toplevels can hold the activated flag at once. The one that
    /// can answer the question wins, whatever order the map iterates in.
    #[test]
    fn a_nameless_activated_toplevel_never_shadows_a_named_one() {
        for _ in 0..32 {
            let mut t = Toplevels::default();
            t.track(1);
            t.set_state(1, &wl_array(&[STATE_ACTIVATED]));
            t.track(2);
            t.set_app_id(2, "kitty".into());
            t.set_state(2, &wl_array(&[STATE_ACTIVATED]));
            assert_eq!(t.activated_app_id().as_deref(), Some("kitty"));
        }
    }

    #[test]
    fn a_closed_toplevel_is_forgotten() {
        let mut t = Toplevels::default();
        t.track(1);
        t.set_app_id(1, "kitty".into());
        t.set_state(1, &wl_array(&[STATE_ACTIVATED]));
        t.forget(1);
        assert_eq!(t.activated_app_id(), None);
    }

    #[test]
    fn the_burst_is_over_once_every_toplevel_is_done() {
        let mut t = Toplevels::default();
        assert!(t.all_done(), "nothing announced yet");
        t.track(1);
        t.track(2);
        assert!(!t.all_done());
        t.set_done(1);
        assert!(!t.all_done());
        t.set_done(2);
        assert!(t.all_done());
    }

    /// A handle can speak before the manager's announcement is processed; the
    /// record is created on demand rather than dropping the event.
    #[test]
    fn an_untracked_handle_still_records_its_events() {
        let mut t = Toplevels::default();
        t.set_app_id(7, "konsole".into());
        t.set_state(7, &wl_array(&[STATE_ACTIVATED]));
        assert_eq!(t.activated_app_id().as_deref(), Some("konsole"));
    }
}
