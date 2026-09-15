// SPDX-License-Identifier: GPL-3.0-or-later

//! Linux-only global hotkey detection via evdev with device grab.
//!
//! Grabs physical keyboards exclusively and re-injects non-hotkey events
//! through a uinput passthrough device. This works on both X11 and Wayland.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, mpsc as std_mpsc};
use std::time::Duration;

use evdev::{
    AttributeSet, Device, EventType, InputEvent, KeyCode, RelativeAxisCode, uinput::VirtualDevice,
};
use tracing::{debug, error, info, warn};

use lune_core::config::{ActivationMode, HotkeyConfig, Modifier};

use crate::hotkey::{HotkeyAction, HotkeyBackend, HotkeySink};

/// Live-updatable hotkey configuration shared between the settings-save
/// path and the evdev listener thread. The listener polls `generation` per
/// key event (a single atomic load) and re-reads the snapshot only when it
/// changed, so every hotkey setting — trigger key, modifiers, cancel key,
/// activation mode — applies immediately without restarting the listener.
pub struct LiveHotkeyConfig {
    inner: RwLock<HotkeySnapshot>,
    generation: AtomicU64,
    /// Cleared when the listener thread exits, so the controller can stop
    /// re-targeting a backend that no longer reads the keyboard.
    alive: Arc<AtomicBool>,
}

#[derive(Clone)]
struct HotkeySnapshot {
    main_trigger: KeyCode,
    main_mods: Vec<Modifier>,
    cancel_trigger: KeyCode,
    cancel_mods: Vec<Modifier>,
    mode: ActivationMode,
}

impl HotkeySnapshot {
    fn from_config(config: &HotkeyConfig, mode: ActivationMode) -> Option<Self> {
        let main_trigger = key_name_to_code(&config.toggle_dictation.key)?;
        let cancel_trigger = key_name_to_code(&config.cancel_dictation.key).unwrap_or_else(|| {
            warn!(
                key = %config.cancel_dictation.key,
                "unknown cancel hotkey key; cancel hotkey disabled"
            );
            KeyCode::KEY_RESERVED
        });
        Some(Self {
            main_trigger,
            main_mods: config.toggle_dictation.modifiers.clone(),
            cancel_trigger,
            cancel_mods: config.cancel_dictation.modifiers.clone(),
            mode,
        })
    }
}

impl LiveHotkeyConfig {
    /// Returns `None` when the trigger key name is unknown.
    pub fn new(config: &HotkeyConfig, mode: ActivationMode) -> Option<Self> {
        Some(Self {
            inner: RwLock::new(HotkeySnapshot::from_config(config, mode)?),
            generation: AtomicU64::new(0),
            alive: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Swap in a new configuration. On an unknown trigger key the previous
    /// config is kept and an error returned — a half-applied hotkey is worse
    /// than a stale one.
    pub fn update(&self, config: &HotkeyConfig, mode: ActivationMode) -> Result<(), String> {
        let snap = HotkeySnapshot::from_config(config, mode)
            .ok_or_else(|| format!("unknown hotkey key name: {}", config.toggle_dictation.key))?;
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = snap;
        self.generation.fetch_add(1, Ordering::Release);
        info!(
            key = %config.toggle_dictation.key,
            ?mode,
            "hotkey configuration hot-reloaded"
        );
        Ok(())
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn snapshot(&self) -> HotkeySnapshot {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl HotkeyBackend for LiveHotkeyConfig {
    fn update(&self, config: &HotkeyConfig, mode: ActivationMode) -> Result<(), String> {
        LiveHotkeyConfig::update(self, config, mode)
    }

    fn name(&self) -> &'static str {
        "evdev"
    }

    fn is_live(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
}

/// Start the evdev listener. The `hotkey::start` entry point for Linux.
///
/// The listener thread runs for the process lifetime, so its join handle is
/// deliberately dropped: there is no shutdown path that could use it, and the
/// grab must outlive every caller.
pub fn start(
    config: &HotkeyConfig,
    mode: ActivationMode,
    sink: HotkeySink,
) -> Option<Arc<dyn HotkeyBackend>> {
    let live = Arc::new(LiveHotkeyConfig::new(config, mode)?);
    spawn(Arc::clone(&live), sink)?;
    Some(live)
}

/// How long to wait for the listener to say whether it opened any keyboards.
///
/// Enumeration is a directory read and a few ioctls, so this is far more than
/// it needs. Being wrong costs one launch falling back to the toggle-only
/// plugin, which the next settings save repairs.
const LISTENER_READY_TIMEOUT: Duration = Duration::from_secs(2);

/// Spawn the evdev hotkey listener on a background OS thread.
///
/// Returns `None` if the hotkey config references unknown key names or no
/// keyboards are accessible.
///
/// Waits for the listener to confirm it has devices rather than reporting
/// success the moment the thread exists. A thread that spawns and then finds
/// nothing to grab — the usual AppImage/tarball case before a reboot — would
/// otherwise be indistinguishable from a working one: the caller skips the
/// global-shortcut fallback, `HotkeyController::update` re-targets a listener
/// that is not there and returns `Ok(())`, and every UI affordance reports a
/// hotkey that can never fire.
fn spawn(live: Arc<LiveHotkeyConfig>, sink: HotkeySink) -> Option<std::thread::JoinHandle<()>> {
    let (ready_tx, ready_rx) = std_mpsc::sync_channel::<bool>(1);
    let alive = Arc::clone(&live.alive);
    let handle = std::thread::Builder::new()
        .name("hotkey-evdev".into())
        .spawn(move || {
            run_listener(live, sink, ready_tx);
            // However the loop ended — a dead sink, no devices — this backend
            // no longer reads the keyboard, and the controller has to know.
            alive.store(false, Ordering::Release);
            warn!("evdev hotkey listener stopped");
        })
        .map_err(|e| warn!(error = %e, "failed to spawn hotkey listener thread"))
        .ok()?;

    match ready_rx.recv_timeout(LISTENER_READY_TIMEOUT) {
        Ok(true) => Some(handle),
        Ok(false) => {
            warn!("evdev listener found no grabbable keyboard; falling back to global shortcuts");
            None
        }
        Err(std_mpsc::RecvTimeoutError::Disconnected) => {
            warn!(
                "evdev listener thread ended before reporting readiness; falling back to global shortcuts"
            );
            None
        }
        // The thread is still inside `open_devices`, which grabs as it goes.
        // Reporting failure here would register the global-shortcut fallback
        // on top of a listener that is about to own every keyboard: one press
        // would fire twice, and nothing can stop the orphan. A slow
        // enumeration is the lesser problem, so adopt it.
        Err(std_mpsc::RecvTimeoutError::Timeout) => {
            warn!(
                timeout_s = LISTENER_READY_TIMEOUT.as_secs(),
                "evdev listener is still enumerating; adopting it rather than racing a fallback"
            );
            Some(handle)
        }
    }
}

// ─── Listener core ───────────────────────────────────────────────────────────

fn build_combo_mods(mods: &[Modifier]) -> PressedCodes {
    let mut combo = PressedCodes::default();
    for m in mods {
        let (l, r) = modifier_keycodes(*m);
        combo.insert(l);
        combo.insert(r);
    }
    combo
}

/// How long the key stream must be quiet before a withheld modifier press is
/// released to apps.
///
/// A deferred modifier is a bet that the next event completes the hotkey. The
/// bet has to expire: reaching for the mouse with Ctrl held produces no key
/// event at all, so an unbounded defer means Ctrl+click and Ctrl+scroll simply
/// never receive their modifier. Long enough that a real chord still resolves
/// through the fast path, short enough to feel instant.
const IDLE_TICK: Duration = Duration::from_millis(90);

/// Idle ticks between kernel-truth sweeps — roughly half a second of quiet.
const SWEEP_EVERY: u32 = 5;

/// Idle ticks that must pass before the listener looks for new keyboards —
/// roughly two seconds. It also rate-limits respawning: a node that dies the
/// moment it is opened can cost at most one reader thread per interval.
const RESCAN_COOLDOWN_TICKS: u32 = 22;

/// What a reader thread can tell the listener.
enum DeviceEvent {
    /// A raw event, and whether its device was successfully grabbed.
    Input(InputEvent, bool),
    /// This node is no longer being read. The listener drops it so a rescan
    /// will grab it again if it comes back — without this a keyboard that
    /// sleeps, or a receiver that re-enumerates, is ignored until restart.
    Gone(PathBuf),
}

/// Everything the listener loop needs from the OS.
struct ListenerIo {
    /// Nodes that currently have a reader thread.
    watched: std::collections::HashSet<PathBuf>,
    /// Second, ungrabbed handles on the watched nodes. They are never read
    /// from — they exist only to answer `EVIOCGKEY`, the kernel's record of
    /// which keys are physically held. That is the one source of truth the
    /// event stream cannot lie about, and `sweep` repairs drift against it.
    probes: Vec<(PathBuf, Device)>,
    fwd: Forwarder<VirtualDevice>,
    /// Kept alive by the listener so the channel outlives every reader: with
    /// no keyboard attached the loop must keep ticking, not exit.
    tx: std_mpsc::Sender<DeviceEvent>,
    rx: std_mpsc::Receiver<DeviceEvent>,
}

fn open_devices() -> Option<ListenerIo> {
    let passthrough = match build_passthrough_device() {
        Ok(dev) => dev,
        Err(e) => {
            warn!(error = %e, "failed to create passthrough device; hotkey grab disabled");
            return None;
        }
    };

    let (tx, rx) = std_mpsc::channel::<DeviceEvent>();
    let mut io = ListenerIo {
        watched: std::collections::HashSet::new(),
        probes: Vec::new(),
        fwd: Forwarder::new(passthrough),
        tx,
        rx,
    };
    if attach_new_keyboards(&mut io) == 0 {
        // Not fatal any more: a Bluetooth keyboard that pairs later, or one
        // plugged in after launch, is picked up by the rescan.
        warn!("no accessible keyboard devices found; waiting for one to appear");
    }
    Some(io)
}

/// Grab every keyboard that has no reader thread yet. Returns how many were
/// added. Used for the initial attach and for every rescan, so a keyboard
/// present at launch and one plugged in later take the same path.
fn attach_new_keyboards(io: &mut ListenerIo) -> usize {
    let mut added = 0;
    for (path, dev) in enumerate_keyboards() {
        if !io.watched.insert(path.clone()) {
            continue;
        }
        open_probe(io, &path);
        let name = dev.name().unwrap_or("unknown").to_string();
        let tx = io.tx.clone();
        let reader_path = path.clone();
        match std::thread::Builder::new()
            .name(format!("evdev-{name}"))
            .spawn(move || read_device(reader_path, dev, tx))
        {
            Ok(_) => added += 1,
            Err(e) => {
                warn!(error = %e, device = %name, "failed to spawn evdev reader thread");
                io.watched.remove(&path);
                io.probes.retain(|(p, _)| p != &path);
            }
        }
    }
    if added > 0 {
        info!(added, watched = io.watched.len(), "keyboards attached");
    }
    added
}

/// Open the second, ungrabbed handle a node needs to answer `EVIOCGKEY`.
///
/// Worth reporting when it fails: without one, `physical_keys` has no truth to
/// compare against and the sweep stands down entirely, so the drift repair is
/// off until the handle opens.
fn open_probe(io: &mut ListenerIo, path: &PathBuf) {
    match Device::open(path) {
        Ok(probe) => io.probes.push((path.clone(), probe)),
        Err(e) => warn!(
            path = %path.display(),
            error = %e,
            "no probe handle for this keyboard; drift repair is off until it opens"
        ),
    }
}

/// Retry the probe handles that never opened. The reader can win the race with
/// udev's ACL on a freshly plugged keyboard, leaving a watched node with no
/// probe; the node is already in `watched`, so only this reopens it — otherwise
/// the sweep stays disabled until the device is physically reconnected.
fn retry_missing_probes(io: &mut ListenerIo) {
    let missing: Vec<PathBuf> = io
        .watched
        .iter()
        .filter(|path| !io.probes.iter().any(|(probed, _)| probed == *path))
        .cloned()
        .collect();
    for path in missing {
        open_probe(io, &path);
    }
}

/// Whether to go looking for keyboards again.
///
/// Never more often than the cooldown, and then only when something actually
/// changed: a reader thread reported its device gone, or the set of nodes in
/// `/dev/input` moved. A device removed and re-added under the same node name
/// leaves the directory looking identical, which is why the dead reader is a
/// signal in its own right.
fn rescan_due(ticks_since_rescan: u32, reader_lost: bool, nodes_changed: bool) -> bool {
    ticks_since_rescan >= RESCAN_COOLDOWN_TICKS && (reader_lost || nodes_changed)
}

/// A cheap fingerprint of `/dev/input`: the sorted event-node names. Comparing
/// it costs one directory read, against re-opening and interrogating every
/// device, and it changes exactly when a node appears or disappears.
fn input_node_signature() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir("/dev/input") else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("event"))
        .collect();
    names.sort_unstable();
    names
}

fn run_listener(live: Arc<LiveHotkeyConfig>, sink: HotkeySink, ready: std_mpsc::SyncSender<bool>) {
    let mut io = match open_devices() {
        Some(io) => {
            let _ = ready.send(true);
            io
        }
        None => {
            let _ = ready.send(false);
            return;
        }
    };

    let mut config_gen = live.generation();
    let mut snap = live.snapshot();
    let mut combo_mod_codes = build_combo_mods(&snap.main_mods);

    let mut state = HotkeyState::new(
        snap.main_trigger,
        snap.main_mods.clone(),
        snap.cancel_trigger,
        snap.cancel_mods.clone(),
    );
    let mut consumed = PressedCodes::default();
    let mut deferred: Vec<InputEvent> = Vec::new();

    let mut idle_ticks: u32 = 0;
    let mut ticks_since_rescan: u32 = 0;
    let mut reader_lost = false;
    let mut sweep_soon = false;
    let mut node_sig = input_node_signature();

    loop {
        let (raw, grabbed) = match io.rx.recv_timeout(IDLE_TICK) {
            Ok(DeviceEvent::Input(raw, grabbed)) => {
                idle_ticks = 0;
                (raw, grabbed)
            }
            Ok(DeviceEvent::Gone(path)) => {
                info!(path = %path.display(), "keyboard detached");
                io.watched.remove(&path);
                io.probes.retain(|(p, _)| p != &path);
                // Whatever it was holding will never send a release, and its
                // probe can no longer vouch for anything. Sweep at the next
                // opportunity rather than waiting out the usual interval.
                sweep_soon = true;
                reader_lost = true;
                continue;
            }
            Err(std_mpsc::RecvTimeoutError::Timeout) => {
                idle_ticks += 1;
                ticks_since_rescan = ticks_since_rescan.saturating_add(1);

                let due = sweep_soon || idle_ticks >= SWEEP_EVERY;
                if due {
                    idle_ticks = 0;
                    sweep_soon = false;
                }
                if let Some(action) = on_idle(
                    &mut io.fwd,
                    &mut state,
                    &mut consumed,
                    &mut deferred,
                    Probes {
                        handles: &io.probes,
                        watched: io.watched.len(),
                    },
                    snap.mode,
                    due,
                ) {
                    sink.emit(action);
                }
                if io.fwd.sink_failed() {
                    break;
                }

                // Cheap gate: only touch the filesystem once the cooldown has
                // elapsed, and only re-enumerate when something moved.
                if ticks_since_rescan >= RESCAN_COOLDOWN_TICKS {
                    // A watched node whose probe never opened is checked every
                    // cooldown, not just when the device set moves: the ACL it
                    // lost the race to is granted without any node changing.
                    retry_missing_probes(&mut io);
                    let sig = input_node_signature();
                    if rescan_due(ticks_since_rescan, reader_lost, sig != node_sig) {
                        reader_lost = false;
                        attach_new_keyboards(&mut io);
                    }
                    node_sig = sig;
                    ticks_since_rescan = 0;
                }
                continue;
            }
            // Unreachable while `io.tx` is alive, and it lives as long as this
            // loop — a keyboardless machine must keep waiting, not give up.
            Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
        };

        // Hot-reload: one atomic load per event; rebuild targeting only when
        // a settings save bumped the generation. Pressed-key tracking is
        // preserved so a mid-combo reload can't wedge the state machine.
        let current_gen = live.generation();
        if current_gen != config_gen {
            config_gen = current_gen;
            snap = live.snapshot();
            combo_mod_codes = build_combo_mods(&snap.main_mods);
            state.retarget(
                snap.main_trigger,
                snap.main_mods.clone(),
                snap.cancel_trigger,
                snap.cancel_mods.clone(),
            );
        }

        if let Some(action) = handle_event(
            raw,
            grabbed,
            &mut state,
            &mut io.fwd,
            &mut consumed,
            &mut deferred,
            &combo_mod_codes,
            snap.mode,
        ) {
            sink.emit(action);
        }
        if io.fwd.sink_failed() {
            break;
        }
    }
    if io.fwd.sink_failed() {
        // Nothing can be written any more, so the synthetic releases below are
        // a formality. What matters is dropping `io`: the reader threads lose
        // their channel, return, and drop the `Device` handles — and closing
        // those fds is what releases the grabs and gives the user a keyboard
        // back.
        error!(
            "uinput passthrough is dead; releasing the keyboard grabs and stopping the listener"
        );
    }
    io.fwd.release_all();
    info!("hotkey listener exiting");
}

/// Route one raw event from a keyboard: forward what apps must still see, and
/// return the hotkey action it produced, if any.
#[allow(clippy::too_many_arguments)]
fn handle_event<S: KeySink>(
    raw: InputEvent,
    grabbed: bool,
    state: &mut HotkeyState,
    fwd: &mut Forwarder<S>,
    consumed: &mut PressedCodes,
    deferred: &mut Vec<InputEvent>,
    combo_mod_codes: &PressedCodes,
    mode: ActivationMode,
) -> Option<HotkeyAction> {
    if raw.event_type() != EventType::KEY {
        if grabbed {
            fwd.forward(raw);
        }
        return None;
    }

    let code = KeyCode(raw.code());
    let value = raw.value();

    // Auto-repeat
    if value == 2 {
        if grabbed && !consumed.contains(code) && !combo_mod_codes.contains(code) {
            fwd.forward(raw);
        }
        return None;
    }

    let pressed = value == 1;

    // A device we could not grab still reaches its apps through the kernel, so
    // it drives the state machine and nothing else.
    if !grabbed {
        return state.handle(code, pressed, mode);
    }

    process_grabbed_key(
        raw,
        code,
        pressed,
        state,
        fwd,
        consumed,
        deferred,
        combo_mod_codes,
        mode,
    )
}
/// Work done when the key stream falls quiet: resolve any withheld modifier
/// press, and — once every `SWEEP_EVERY` ticks — repair drift against the
/// kernel. Split out of the listener loop because both halves are only ever
/// correct in the absence of pending events.
fn on_idle<S: KeySink>(
    fwd: &mut Forwarder<S>,
    state: &mut HotkeyState,
    consumed: &mut PressedCodes,
    deferred: &mut Vec<InputEvent>,
    probes: Probes<'_>,
    mode: ActivationMode,
    sweep_due: bool,
) -> Option<HotkeyAction> {
    // Nothing more is coming to complete a chord, so a withheld modifier press
    // has to be delivered now rather than wait for a key event that may never
    // arrive.
    fwd.flush(deferred);
    if !sweep_due {
        return None;
    }
    sweep(fwd, state, consumed, probes, mode)
}

/// Reconcile what apps believe against what the kernel reports.
///
/// A key release can go missing for reasons this process never observes: a
/// wireless keyboard that sleeps and re-enumerates mid-press, a uinput write
/// that fails, a device node replaced underneath a reader thread. The cost is
/// not a lost keystroke but a key wedged *down* in every application until the
/// user restarts — and, on the state-machine side, a phantom pressed key that
/// `extra_mods` reads as an unwanted modifier, so the hotkey silently stops
/// matching. `EVIOCGKEY` knows the truth in both cases, which turns permanent
/// corruption into drift that heals itself within half a second of quiet.
fn sweep<S: KeySink>(
    fwd: &mut Forwarder<S>,
    state: &mut HotkeyState,
    consumed: &mut PressedCodes,
    probes: Probes<'_>,
    mode: ActivationMode,
) -> Option<HotkeyAction> {
    // No kernel truth means no repair. `reconcile` cannot tell "the kernel says
    // this key is up" from "nobody answered", so acting on a partial answer
    // would release keys the user is still holding — the very damage the sweep
    // exists to undo.
    let Some(physical) = probes.physical_keys() else {
        debug!("kernel key state unavailable; skipping this sweep");
        return None;
    };
    let stray = fwd.reconcile(&physical);
    if !stray.is_empty() {
        warn!(?stray, "released keys the kernel reports as up");
    }
    consumed.retain(&physical);
    state.sync_pressed(&physical, mode)
}

/// The ungrabbed handles the sweep reads kernel truth from, carried together
/// with the number of keyboards they are supposed to cover. The two travel as
/// one because an answer from only some of them is not an answer.
#[derive(Clone, Copy)]
struct Probes<'a> {
    handles: &'a [(PathBuf, Device)],
    watched: usize,
}

impl Probes<'_> {
    /// Which keys the kernel currently reports as physically held, across every
    /// keyboard this listener watches.
    ///
    /// `None` when the answer would be incomplete — a watched node with no
    /// probe handle, or an `EVIOCGKEY` that failed. An incomplete answer is not
    /// the same as "no keys are held": the caller synthesizes a release for
    /// every key missing from this set, so a keyboard that silently contributes
    /// nothing would have its held keys released under the user's fingers.
    fn physical_keys(&self) -> Option<PressedCodes> {
        if self.handles.len() < self.watched {
            return None;
        }
        let mut set = PressedCodes::default();
        for (_, dev) in self.handles {
            for code in dev.get_key_state().ok()?.iter() {
                set.insert(code);
            }
        }
        Some(set)
    }
}

/// A destination for re-injected key events: the real uinput `VirtualDevice`
/// in production, a recorder in tests. `false` means the event was not
/// written.
trait KeySink {
    fn send_event(&mut self, event: InputEvent) -> bool;
}

impl KeySink for VirtualDevice {
    fn send_event(&mut self, event: InputEvent) -> bool {
        if let Err(e) = self.emit(&[event, InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0)]) {
            // A dropped release is a key left down in every app. The idle sweep
            // repairs it; this line is how we learn it happened at all.
            warn!(
                error = %e,
                code = event.code(),
                value = event.value(),
                "passthrough write failed"
            );
            return false;
        }
        true
    }
}

/// Consecutive failed passthrough writes that mean the uinput node itself is
/// gone (ENODEV/EIO) rather than one event being dropped under load.
const SINK_FAILURE_LIMIT: u32 = 10;

/// Wraps the passthrough sink and tracks which key codes were forwarded as
/// currently-down. This guards two invariants:
///   1. A release is never swallowed for a key whose press we forwarded — that
///      would strand the key down in apps (a stuck modifier).
///   2. On listener exit, every still-down forwarded key gets a synthetic
///      release, so a teardown mid-hold can't leave a key wedged.
///   3. A sink that has stopped accepting writes is reported, so the listener
///      can exit and let the reader threads drop their grabs. Grabbed
///      keyboards deliver here alone: with the passthrough dead every key is
///      swallowed in every app, and there is no keyboard left to kill the
///      process with.
struct Forwarder<S: KeySink> {
    sink: S,
    down: PressedCodes,
    /// Failed writes since the last successful one.
    failures: u32,
}

impl<S: KeySink> Forwarder<S> {
    fn new(sink: S) -> Self {
        Self {
            sink,
            down: PressedCodes::default(),
            failures: 0,
        }
    }

    /// Send through the sink, tracking the run of failures.
    fn emit(&mut self, event: InputEvent) {
        if self.sink.send_event(event) {
            self.failures = 0;
        } else {
            self.failures = self.failures.saturating_add(1);
        }
    }

    /// Whether the sink has failed often enough in a row to call it dead.
    fn sink_failed(&self) -> bool {
        self.failures >= SINK_FAILURE_LIMIT
    }

    /// Forward an event, recording key press/release in the forwarded-down set.
    fn forward(&mut self, event: InputEvent) {
        if event.event_type() == EventType::KEY {
            let code = KeyCode(event.code());
            match event.value() {
                1 => self.down.insert(code),
                0 => self.down.remove(code),
                _ => {}
            }
        }
        self.emit(event);
    }

    fn flush(&mut self, deferred: &mut Vec<InputEvent>) {
        for ev in deferred.drain(..) {
            self.forward(ev);
        }
    }

    /// Whether a press for `code` was forwarded without a matching release.
    fn is_down(&self, code: KeyCode) -> bool {
        self.down.contains(code)
    }

    /// Release every key we told apps was down that the kernel says is up, and
    /// report which ones. See `sweep` for why the two can disagree.
    fn reconcile(&mut self, physical: &PressedCodes) -> Vec<KeyCode> {
        let stray: Vec<KeyCode> = self
            .down
            .codes()
            .into_iter()
            .filter(|code| !physical.contains(*code))
            .collect();
        for code in &stray {
            self.forward(InputEvent::new(EventType::KEY.0, code.code(), 0));
        }
        stray
    }

    /// Emit a synthetic release for every still-down forwarded key. Called on
    /// listener exit so a teardown while keys are held can't strand them.
    fn release_all(&mut self) {
        for code in self.down.drain_codes() {
            self.emit(InputEvent::new(EventType::KEY.0, code.code(), 0));
        }
    }
}

/// Handle one non-repeat key event from a grabbed device: drive the state
/// machine and forward or swallow the raw event through `fwd`. Returns the
/// action to dispatch, if any.
#[allow(clippy::too_many_arguments)]
fn process_grabbed_key<S: KeySink>(
    raw: InputEvent,
    code: KeyCode,
    pressed: bool,
    state: &mut HotkeyState,
    fwd: &mut Forwarder<S>,
    consumed: &mut PressedCodes,
    deferred: &mut Vec<InputEvent>,
    combo_mod_codes: &PressedCodes,
    mode: ActivationMode,
) -> Option<HotkeyAction> {
    // Defer combo modifier presses. If the modifier press itself completes a
    // hotkey (trigger key was already down), drop the deferred queue and
    // dispatch the action instead of forwarding the modifier through to apps.
    // Without this, the Start action is silently dropped and the next release
    // emits a stray Stop.
    if pressed && combo_mod_codes.contains(code) {
        deferred.push(raw);
        if let Some(action) = state.handle(code, pressed, mode) {
            deferred.clear();
            consumed.insert(code);
            return Some(action);
        }
        return None;
    }

    match state.handle(code, pressed, mode) {
        Some(HotkeyAction::Cancel) => {
            fwd.flush(deferred);
            fwd.forward(raw);
            Some(HotkeyAction::Cancel)
        }
        Some(a) => {
            deferred.clear();
            if pressed {
                consumed.insert(code);
            } else {
                consumed.remove(code);
                // Keep apps balanced: if this key's press reached them, its
                // release must too, or the action just stranded it down.
                if fwd.is_down(code) {
                    fwd.forward(raw);
                }
            }
            Some(a)
        }
        None if !pressed && consumed.contains(code) => {
            consumed.remove(code);
            if fwd.is_down(code) {
                fwd.forward(raw);
            }
            None
        }
        None if !pressed && combo_mod_codes.contains(code) => {
            // Flush first: a press still sitting in `deferred` is this
            // modifier's own (a bare tap), and apps need it before the release.
            fwd.flush(deferred);
            // A press that was dropped rather than forwarded — the combo
            // completed on the trigger key and cleared the queue — left apps
            // never seeing this modifier go down. Sending the release anyway
            // cancels a modifier the focused app is legitimately holding.
            if fwd.is_down(code) {
                fwd.forward(raw);
            }
            None
        }
        None => {
            fwd.flush(deferred);
            fwd.forward(raw);
            None
        }
    }
}

// ─── Device enumeration ──────────────────────────────────────────────────────

const SELF_PREFIX: &str = "lune-voice";

/// USB vendor/product pair. Every event node a physical device exposes carries
/// the same one, which is how a mouse's keyboard interface can be told from a
/// keyboard: the pointer node sitting beside it shares its id.
type DeviceIdent = (u16, u16);

fn device_ident(dev: &Device) -> DeviceIdent {
    let id = dev.input_id();
    (id.vendor(), id.product())
}

/// Whether this node is a pointer. Gate on pointer *movement* (REL_X) or mouse
/// buttons — NOT any relative axis, or a keyboard's rotary knob / volume wheel
/// (REL_WHEEL / REL_DIAL) would wrongly mark the keyboard a pointer.
fn is_pointer_node(dev: &Device) -> bool {
    dev.supported_keys()
        .is_some_and(|k| k.contains(KeyCode::BTN_LEFT))
        || dev
            .supported_relative_axes()
            .is_some_and(|a| a.contains(RelativeAxisCode::REL_X))
}

fn enumerate_keyboards() -> Vec<(PathBuf, Device)> {
    let all: Vec<(PathBuf, Device)> = evdev::enumerate()
        .filter(|(path, dev)| {
            let is_self = dev.name().is_some_and(|n| n.starts_with(SELF_PREFIX));
            if is_self {
                debug!(path = %path.display(), "skipping self device");
            }
            !is_self
        })
        .collect();

    let pointer_idents: Vec<DeviceIdent> = all
        .iter()
        .filter(|(_, dev)| is_pointer_node(dev))
        .map(|(_, dev)| device_ident(dev))
        .collect();

    all.into_iter()
        .filter(|(path, dev)| {
            let Some(keys) = dev.supported_keys() else {
                return false;
            };
            let typing = is_typing_keyboard_caps(
                keys.contains(KeyCode::KEY_A),
                keys.contains(KeyCode::KEY_SPACE),
                keys.contains(KeyCode::KEY_LEFTSHIFT) || keys.contains(KeyCode::KEY_RIGHTSHIFT),
                keys.contains(KeyCode::KEY_ENTER),
                is_pointer_node(dev),
            );
            if !node_is_grabbable(typing, device_ident(dev), &pointer_idents) {
                debug!(name = ?dev.name(), path = %path.display(), "skipping non-typing-keyboard node");
                return false;
            }
            info!(name = ?dev.name(), path = %path.display(), "found keyboard");
            true
        })
        .collect()
}

/// Whether a node that looks like a typing keyboard should actually be grabbed.
///
/// A gaming mouse's keyboard interface advertises a full keyboard key map and
/// carries no pointer bits of its own, so capabilities alone cannot tell the
/// two apart — but its sibling pointer node can, because both nodes report the
/// same vendor/product. Grabbing one is how a wireless mouse's own sleep/wake
/// cycle ends up tearing down a keyboard grab at random.
fn node_is_grabbable(
    is_typing_keyboard: bool,
    ident: DeviceIdent,
    pointer_idents: &[DeviceIdent],
) -> bool {
    if !is_typing_keyboard {
        return false;
    }
    // `(0, 0)` is the unset id shared by unrelated virtual devices. It
    // identifies nothing, so it must not drag a real keyboard down with a
    // stranger's mouse.
    ident == (0, 0) || !pointer_idents.contains(&ident)
}

/// Whether a device's capabilities mark it a real typing keyboard: letters,
/// space, a Shift, and Enter, and not a pointer. Pure so the rule is
/// unit-tested. The old filter accepted any device exposing `A`+`Space`, which
/// also matched a QMK board's consumer-control node; requiring Shift+Enter
/// rejects that. It cannot reject a gaming mouse's keyboard interface, which
/// advertises the full key map — `node_is_grabbable` does that instead.
///
/// A board that exposes two genuine typing nodes (a QMK keyboard's 6KRO and
/// NKRO interfaces, say) is still grabbed on both, deliberately: the firmware
/// chooses between them at runtime, so watching only one drops keys.
fn is_typing_keyboard_caps(
    has_a: bool,
    has_space: bool,
    has_shift: bool,
    has_enter: bool,
    is_pointer: bool,
) -> bool {
    has_a && has_space && has_shift && has_enter && !is_pointer
}

fn read_device(path: PathBuf, mut dev: Device, tx: std_mpsc::Sender<DeviceEvent>) {
    let name = dev.name().unwrap_or("unknown").to_string();
    let grabbed = match dev.grab() {
        Ok(()) => {
            info!(name, "grabbed keyboard");
            true
        }
        Err(e) => {
            debug!(error = %e, name, "grab failed; observe-only");
            false
        }
    };
    loop {
        match dev.fetch_events() {
            Ok(events) => {
                for ev in events {
                    if tx.send(DeviceEvent::Input(ev, grabbed)).is_err() {
                        // The listener is gone; there is nobody left to tell.
                        return;
                    }
                }
            }
            Err(e) => {
                debug!(error = %e, name, "device disconnected");
                break;
            }
        }
    }
    // Hand the node back, so the same keyboard returning — or a replacement on
    // the same path — is grabbed again instead of ignored until restart.
    let _ = tx.send(DeviceEvent::Gone(path));
}

fn build_passthrough_device() -> Result<VirtualDevice, String> {
    let mut keys = AttributeSet::<KeyCode>::new();
    for i in 1u16..768 {
        keys.insert(KeyCode(i));
    }
    let dev = VirtualDevice::builder()
        .map_err(|e| format!("passthrough builder: {e}"))?
        .name("lune-voice-passthrough")
        .with_keys(&keys)
        .map_err(|e| format!("passthrough with_keys: {e}"))?
        .build()
        .map_err(|e| format!("passthrough build: {e}"))?;
    std::thread::sleep(std::time::Duration::from_millis(200));
    debug!("passthrough virtual keyboard ready");
    Ok(dev)
}

// ─── Hotkey state machine ────────────────────────────────────────────────────

struct HotkeyState {
    pressed: PressedCodes,
    main_trigger: KeyCode,
    main_mods: Vec<Modifier>,
    cancel_trigger: KeyCode,
    cancel_mods: Vec<Modifier>,
    hold_active: bool,
}

impl HotkeyState {
    fn new(
        main_trigger: KeyCode,
        main_mods: Vec<Modifier>,
        cancel_trigger: KeyCode,
        cancel_mods: Vec<Modifier>,
    ) -> Self {
        Self {
            pressed: PressedCodes::default(),
            main_trigger,
            main_mods,
            cancel_trigger,
            cancel_mods,
            hold_active: false,
        }
    }

    /// Swap key/modifier targets in place, preserving the physical
    /// pressed-key set. `hold_active` resets — a hold in progress under the
    /// old combo must not emit a Stop for the new one.
    fn retarget(
        &mut self,
        main_trigger: KeyCode,
        main_mods: Vec<Modifier>,
        cancel_trigger: KeyCode,
        cancel_mods: Vec<Modifier>,
    ) {
        self.main_trigger = main_trigger;
        self.main_mods = main_mods;
        self.cancel_trigger = cancel_trigger;
        self.cancel_mods = cancel_mods;
        self.hold_active = false;
    }

    /// Replace the tracked pressed-key set with the kernel's ground truth.
    ///
    /// Returns `Stop` when a hold's keys are no longer held: without it a lost
    /// release keeps the microphone open on a session the user believes they
    /// ended. A stale key is equally damaging when idle — `extra_mods` counts
    /// it as an unwanted modifier, so the hotkey stops matching entirely.
    fn sync_pressed(
        &mut self,
        physical: &PressedCodes,
        mode: ActivationMode,
    ) -> Option<HotkeyAction> {
        self.pressed = physical.clone();
        if mode != ActivationMode::Hold {
            return None;
        }
        let was_hold = self.hold_active;
        self.hold_active = self.pressed.contains(self.main_trigger)
            && self.mods_down(&self.main_mods)
            && !self.extra_mods(&self.main_mods);
        (was_hold && !self.hold_active).then_some(HotkeyAction::Stop)
    }

    fn handle(
        &mut self,
        code: KeyCode,
        pressed: bool,
        mode: ActivationMode,
    ) -> Option<HotkeyAction> {
        let was_trigger = self.pressed.contains(self.main_trigger);
        let was_hold = self.hold_active;

        if pressed {
            self.pressed.insert(code);
        } else {
            self.pressed.remove(code);
        }

        // Cancel (Escape) in any activation mode. A session can be started by
        // clicking the pill — which has no key to release — so Escape must be
        // able to stop it even in Hold mode. Emitting Cancel here is harmless
        // when idle: the dispatcher no-ops unless a session is active, and the
        // raw Escape is still forwarded on to apps either way.
        if pressed
            && code == self.cancel_trigger
            && self.cancel_trigger != KeyCode::KEY_RESERVED
            && self.mods_down(&self.cancel_mods)
            && !self.extra_mods(&self.cancel_mods)
        {
            return Some(HotkeyAction::Cancel);
        }

        // Main
        let trigger = self.pressed.contains(self.main_trigger);
        let exact = self.mods_down(&self.main_mods) && !self.extra_mods(&self.main_mods);

        match mode {
            ActivationMode::Toggle => {
                if !was_trigger && trigger && exact {
                    return Some(HotkeyAction::Toggle);
                }
            }
            ActivationMode::Hold => {
                self.hold_active = trigger && exact;
                if !was_hold && self.hold_active {
                    return Some(HotkeyAction::Start);
                }
                if was_hold && !self.hold_active {
                    return Some(HotkeyAction::Stop);
                }
            }
            _ => {
                if !was_trigger && trigger && exact {
                    return Some(HotkeyAction::Toggle);
                }
            }
        }
        None
    }

    fn mods_down(&self, mods: &[Modifier]) -> bool {
        mods.iter().all(|m| {
            let (l, r) = modifier_keycodes(*m);
            self.pressed.contains(l) || self.pressed.contains(r)
        })
    }

    fn extra_mods(&self, required: &[Modifier]) -> bool {
        for m in &[
            Modifier::Ctrl,
            Modifier::Shift,
            Modifier::Alt,
            Modifier::Meta,
        ] {
            if required.contains(m) {
                continue;
            }
            let (l, r) = modifier_keycodes(*m);
            if self.pressed.contains(l) || self.pressed.contains(r) {
                return true;
            }
        }
        false
    }
}

// ─── Key mapping ─────────────────────────────────────────────────────────────

fn modifier_keycodes(m: Modifier) -> (KeyCode, KeyCode) {
    match m {
        Modifier::Ctrl => (KeyCode::KEY_LEFTCTRL, KeyCode::KEY_RIGHTCTRL),
        Modifier::Shift => (KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_RIGHTSHIFT),
        Modifier::Alt => (KeyCode::KEY_LEFTALT, KeyCode::KEY_RIGHTALT),
        Modifier::Meta => (KeyCode::KEY_LEFTMETA, KeyCode::KEY_RIGHTMETA),
        _ => (KeyCode::KEY_RESERVED, KeyCode::KEY_RESERVED),
    }
}

fn key_name_to_code(name: &str) -> Option<KeyCode> {
    Some(match name.to_ascii_lowercase().as_str() {
        "space" => KeyCode::KEY_SPACE,
        "escape" | "esc" => KeyCode::KEY_ESC,
        "return" | "enter" => KeyCode::KEY_ENTER,
        "tab" => KeyCode::KEY_TAB,
        "backspace" => KeyCode::KEY_BACKSPACE,
        "delete" => KeyCode::KEY_DELETE,
        "insert" => KeyCode::KEY_INSERT,
        "home" => KeyCode::KEY_HOME,
        "end" => KeyCode::KEY_END,
        "pageup" => KeyCode::KEY_PAGEUP,
        "pagedown" => KeyCode::KEY_PAGEDOWN,
        "up" => KeyCode::KEY_UP,
        "down" => KeyCode::KEY_DOWN,
        "left" => KeyCode::KEY_LEFT,
        "right" => KeyCode::KEY_RIGHT,
        "capslock" => KeyCode::KEY_CAPSLOCK,
        "pause" => KeyCode::KEY_PAUSE,
        "printscreen" | "sysrq" => KeyCode::KEY_SYSRQ,
        "f1" => KeyCode::KEY_F1,
        "f2" => KeyCode::KEY_F2,
        "f3" => KeyCode::KEY_F3,
        "f4" => KeyCode::KEY_F4,
        "f5" => KeyCode::KEY_F5,
        "f6" => KeyCode::KEY_F6,
        "f7" => KeyCode::KEY_F7,
        "f8" => KeyCode::KEY_F8,
        "f9" => KeyCode::KEY_F9,
        "f10" => KeyCode::KEY_F10,
        "f11" => KeyCode::KEY_F11,
        "f12" => KeyCode::KEY_F12,
        "a" => KeyCode::KEY_A,
        "b" => KeyCode::KEY_B,
        "c" => KeyCode::KEY_C,
        "d" => KeyCode::KEY_D,
        "e" => KeyCode::KEY_E,
        "f" => KeyCode::KEY_F,
        "g" => KeyCode::KEY_G,
        "h" => KeyCode::KEY_H,
        "i" => KeyCode::KEY_I,
        "j" => KeyCode::KEY_J,
        "k" => KeyCode::KEY_K,
        "l" => KeyCode::KEY_L,
        "m" => KeyCode::KEY_M,
        "n" => KeyCode::KEY_N,
        "o" => KeyCode::KEY_O,
        "p" => KeyCode::KEY_P,
        "q" => KeyCode::KEY_Q,
        "r" => KeyCode::KEY_R,
        "s" => KeyCode::KEY_S,
        "t" => KeyCode::KEY_T,
        "u" => KeyCode::KEY_U,
        "v" => KeyCode::KEY_V,
        "w" => KeyCode::KEY_W,
        "x" => KeyCode::KEY_X,
        "y" => KeyCode::KEY_Y,
        "z" => KeyCode::KEY_Z,
        "0" => KeyCode::KEY_0,
        "1" => KeyCode::KEY_1,
        "2" => KeyCode::KEY_2,
        "3" => KeyCode::KEY_3,
        "4" => KeyCode::KEY_4,
        "5" => KeyCode::KEY_5,
        "6" => KeyCode::KEY_6,
        "7" => KeyCode::KEY_7,
        "8" => KeyCode::KEY_8,
        "9" => KeyCode::KEY_9,
        "minus" => KeyCode::KEY_MINUS,
        "equal" | "equals" => KeyCode::KEY_EQUAL,
        "leftbrace" => KeyCode::KEY_LEFTBRACE,
        "rightbrace" => KeyCode::KEY_RIGHTBRACE,
        "backslash" => KeyCode::KEY_BACKSLASH,
        "semicolon" => KeyCode::KEY_SEMICOLON,
        "apostrophe" | "quote" => KeyCode::KEY_APOSTROPHE,
        "grave" | "backtick" => KeyCode::KEY_GRAVE,
        "comma" => KeyCode::KEY_COMMA,
        "dot" | "period" => KeyCode::KEY_DOT,
        "slash" => KeyCode::KEY_SLASH,
        _ => return None,
    })
}

// ─── Pressed-key bitset ──────────────────────────────────────────────────────

const LIMIT: usize = 1024;
const WORDS: usize = LIMIT / 64;

#[derive(Clone)]
struct PressedCodes {
    w: [u64; WORDS],
}

impl Default for PressedCodes {
    fn default() -> Self {
        Self { w: [0; WORDS] }
    }
}

impl PressedCodes {
    fn insert(&mut self, c: KeyCode) {
        let i = c.code() as usize;
        if i < LIMIT {
            self.w[i / 64] |= 1u64 << (i % 64);
        }
    }
    fn remove(&mut self, c: KeyCode) {
        let i = c.code() as usize;
        if i < LIMIT {
            self.w[i / 64] &= !(1u64 << (i % 64));
        }
    }
    fn contains(&self, c: KeyCode) -> bool {
        let i = c.code() as usize;
        i < LIMIT && (self.w[i / 64] >> (i % 64)) & 1 != 0
    }
    /// Keep only the codes also set in `keep`.
    fn retain(&mut self, keep: &PressedCodes) {
        for (word, mask) in self.w.iter_mut().zip(keep.w.iter()) {
            *word &= *mask;
        }
    }
    /// Every code currently set, in ascending order.
    fn codes(&self) -> Vec<KeyCode> {
        let mut codes = Vec::new();
        for (wi, word) in self.w.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                codes.push(KeyCode((wi * 64 + b) as u16));
            }
        }
        codes
    }
    /// Drain all set codes, clearing the set. Used to release every key still
    /// marked down on listener exit.
    fn drain_codes(&mut self) -> Vec<KeyCode> {
        let codes = self.codes();
        self.w = [0; WORDS];
        codes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(code: KeyCode, pressed: bool) -> InputEvent {
        InputEvent::new(EventType::KEY.0, code.code(), if pressed { 1 } else { 0 })
    }

    /// A `KeySink` that records the (code, value) of every event it receives.
    #[derive(Default)]
    struct Recorder {
        events: Vec<(u16, i32)>,
    }

    impl KeySink for Recorder {
        fn send_event(&mut self, event: InputEvent) -> bool {
            self.events.push((event.code(), event.value()));
            true
        }
    }

    /// A sink whose writes fail, as a uinput node that has gone ENODEV does.
    #[derive(Default)]
    struct BrokenSink {
        fail: bool,
    }

    impl KeySink for BrokenSink {
        fn send_event(&mut self, _event: InputEvent) -> bool {
            !self.fail
        }
    }

    /// Grabbed keyboards deliver to this process alone, so a sink that stops
    /// writing drops every key in every app — with no keyboard left to kill
    /// the app with. Past a run of failures the listener has to give the
    /// keyboards back.
    #[test]
    fn a_run_of_failed_writes_marks_the_sink_broken() {
        let mut fwd = Forwarder::new(BrokenSink::default());
        assert!(!fwd.sink_failed());

        fwd.sink.fail = true;
        for _ in 0..SINK_FAILURE_LIMIT - 1 {
            fwd.forward(ev(KeyCode::KEY_A, true));
            assert!(!fwd.sink_failed(), "one short of the limit is not broken");
        }
        fwd.forward(ev(KeyCode::KEY_A, true));
        assert!(fwd.sink_failed());
    }

    /// A single dropped event is normal under load. Only an unbroken run
    /// means the node itself is gone.
    #[test]
    fn one_successful_write_clears_the_failure_run() {
        let mut fwd = Forwarder::new(BrokenSink::default());
        fwd.sink.fail = true;
        for _ in 0..SINK_FAILURE_LIMIT - 1 {
            fwd.forward(ev(KeyCode::KEY_A, true));
        }
        fwd.sink.fail = false;
        fwd.forward(ev(KeyCode::KEY_A, true));

        fwd.sink.fail = true;
        for _ in 0..SINK_FAILURE_LIMIT - 1 {
            fwd.forward(ev(KeyCode::KEY_A, true));
        }
        assert!(!fwd.sink_failed(), "the run restarted after the success");
    }

    fn ctrl_space_state() -> HotkeyState {
        HotkeyState::new(
            KeyCode::KEY_SPACE,
            vec![Modifier::Ctrl],
            KeyCode::KEY_ESC,
            vec![],
        )
    }

    #[test]
    fn live_config_update_bumps_generation_and_rejects_unknown_keys() {
        let config = HotkeyConfig::default();
        let live = LiveHotkeyConfig::new(&config, ActivationMode::Toggle).unwrap();
        let g0 = live.generation();
        live.update(&config, ActivationMode::Hold).unwrap();
        assert!(live.generation() > g0);
        assert_eq!(live.snapshot().mode, ActivationMode::Hold);

        let mut bad = config.clone();
        bad.toggle_dictation.key = "definitely-not-a-key".into();
        assert!(live.update(&bad, ActivationMode::Toggle).is_err());
        // Failed update keeps the previous config.
        assert_eq!(live.snapshot().mode, ActivationMode::Hold);
    }

    #[test]
    fn retarget_preserves_pressed_keys_and_clears_hold() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Hold;
        s.handle(KeyCode::KEY_LEFTCTRL, true, m);
        s.handle(KeyCode::KEY_SPACE, true, m); // Start; hold_active = true
        s.retarget(KeyCode::KEY_F9, vec![], KeyCode::KEY_ESC, vec![]);
        // Old combo's release must not emit a Stop for the new combo.
        assert!(s.handle(KeyCode::KEY_SPACE, false, m).is_none());
        // Still-held Ctrl counts as an extra modifier for the bare-F9 combo —
        // pressed-key tracking survived the retarget.
        assert!(s.handle(KeyCode::KEY_F9, true, m).is_none());
        s.handle(KeyCode::KEY_F9, false, m);
        s.handle(KeyCode::KEY_LEFTCTRL, false, m);
        // With Ctrl released, the new trigger works.
        assert!(matches!(
            s.handle(KeyCode::KEY_F9, true, m),
            Some(HotkeyAction::Start)
        ));
    }

    #[test]
    fn hold_mode_emits_start_on_press_and_stop_on_release() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Hold;
        assert!(s.handle(KeyCode::KEY_LEFTCTRL, true, m).is_none());
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, m),
            Some(HotkeyAction::Start)
        ));
        // Releasing the trigger stops, regardless of release order.
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, false, m),
            Some(HotkeyAction::Stop)
        ));
        assert!(s.handle(KeyCode::KEY_LEFTCTRL, false, m).is_none());
    }

    #[test]
    fn hold_mode_stops_when_modifier_released_first() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Hold;
        s.handle(KeyCode::KEY_LEFTCTRL, true, m);
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, m),
            Some(HotkeyAction::Start)
        ));
        assert!(matches!(
            s.handle(KeyCode::KEY_LEFTCTRL, false, m),
            Some(HotkeyAction::Stop)
        ));
    }

    #[test]
    fn mode_switch_mid_session_does_not_panic_or_double_fire() {
        // Listener was in Toggle when the user pressed; settings flip to
        // Hold before release — the release must not emit a spurious action.
        let mut s = ctrl_space_state();
        s.handle(KeyCode::KEY_LEFTCTRL, true, ActivationMode::Toggle);
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, ActivationMode::Toggle),
            Some(HotkeyAction::Toggle)
        ));
        // Hold-mode evaluation on release: hold_active was never set, so the
        // release is silent rather than a stray Stop.
        assert!(
            s.handle(KeyCode::KEY_SPACE, false, ActivationMode::Hold)
                .is_none()
        );
    }

    #[test]
    fn escape_cancels_in_both_activation_modes() {
        // A pill-started session has no held key to release, so the cancel key
        // (Escape) must emit Cancel regardless of activation mode — the
        // dispatcher gates on an active session, not on the mode.
        for mode in [ActivationMode::Toggle, ActivationMode::Hold] {
            let mut s = ctrl_space_state(); // cancel = bare Escape
            assert!(
                matches!(
                    s.handle(KeyCode::KEY_ESC, true, mode),
                    Some(HotkeyAction::Cancel)
                ),
                "Escape should cancel in {mode:?} mode"
            );
        }
    }

    #[test]
    fn typing_keyboard_filter_rejects_secondary_nodes_and_pointers() {
        // A real keyboard: letters, space, shift, enter, not a pointer.
        assert!(is_typing_keyboard_caps(true, true, true, true, false));
        // The Keychron's secondary node (event7): A+Space but no Shift → reject,
        // so the keyboard is no longer grabbed twice.
        assert!(!is_typing_keyboard_caps(true, true, false, true, false));
        // A gaming mouse's macro interface that advertises a full keyboard but
        // carries pointer axes → reject.
        assert!(!is_typing_keyboard_caps(true, true, true, true, true));
        // A media/macro pad without Enter → reject.
        assert!(!is_typing_keyboard_caps(true, true, true, false, false));
    }

    #[test]
    fn forwarded_modifier_release_is_not_swallowed_by_action() {
        // Hold mode, Ctrl+Space. The user holds Ctrl for an unrelated chord
        // (Ctrl+X) — leaking Ctrl-down through to apps — then triggers
        // dictation and releases Ctrl first. The Stop action must not eat the
        // Ctrl release, or Ctrl stays stuck down in every app.
        let m = ActivationMode::Hold;
        let combo = build_combo_mods(&[Modifier::Ctrl]);
        let mut state = ctrl_space_state();
        let mut fwd = Forwarder::new(Recorder::default());
        let mut consumed = PressedCodes::default();
        let mut deferred: Vec<InputEvent> = Vec::new();

        let ctrl = KeyCode::KEY_LEFTCTRL;
        let x = KeyCode::KEY_X;
        let space = KeyCode::KEY_SPACE;

        let step = |code: KeyCode,
                    pressed: bool,
                    state: &mut HotkeyState,
                    fwd: &mut Forwarder<Recorder>,
                    consumed: &mut PressedCodes,
                    deferred: &mut Vec<InputEvent>| {
            process_grabbed_key(
                ev(code, pressed),
                code,
                pressed,
                state,
                fwd,
                consumed,
                deferred,
                &combo,
                m,
            )
        };

        // Ctrl down — deferred, not yet forwarded.
        step(
            ctrl,
            true,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
        );
        // X down — flushes the deferred Ctrl down, then X, to apps.
        step(x, true, &mut state, &mut fwd, &mut consumed, &mut deferred);
        assert!(fwd.is_down(ctrl), "Ctrl press should have reached apps");

        // Space down — completes the combo: Start.
        let start = step(
            space,
            true,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
        );
        assert!(matches!(start, Some(HotkeyAction::Start)));

        // Release Ctrl first — Hold mode emits Stop.
        let stop = step(
            ctrl,
            false,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
        );
        assert!(matches!(stop, Some(HotkeyAction::Stop)));

        // The release must have reached apps, balancing the earlier press.
        assert!(
            !fwd.is_down(ctrl),
            "Ctrl release was swallowed by Stop — Ctrl is stuck down"
        );
    }

    /// Toggle mode, Ctrl+Space, pressed in the ordinary order: Ctrl, then
    /// Space. The Ctrl press is deferred and then dropped when Space completes
    /// the hotkey, so apps never saw Ctrl go down — and must never be told it
    /// came back up. A phantom Ctrl release lands in whatever app has focus and
    /// cancels a modifier it is legitimately holding.
    #[test]
    fn a_swallowed_combo_modifier_is_not_phantom_released() {
        let m = ActivationMode::Toggle;
        let combo = build_combo_mods(&[Modifier::Ctrl]);
        let mut state = ctrl_space_state();
        let mut fwd = Forwarder::new(Recorder::default());
        let mut consumed = PressedCodes::default();
        let mut deferred: Vec<InputEvent> = Vec::new();

        let ctrl = KeyCode::KEY_LEFTCTRL;
        let space = KeyCode::KEY_SPACE;

        let mut step = |code: KeyCode, pressed: bool| {
            process_grabbed_key(
                ev(code, pressed),
                code,
                pressed,
                &mut state,
                &mut fwd,
                &mut consumed,
                &mut deferred,
                &combo,
                m,
            )
        };

        step(ctrl, true);
        assert!(matches!(step(space, true), Some(HotkeyAction::Toggle)));
        step(space, false);
        step(ctrl, false);

        assert!(
            !fwd.sink.events.contains(&(ctrl.code(), 0)),
            "apps never saw Ctrl go down, so they must not be told it went up: {:?}",
            fwd.sink.events
        );
    }

    fn codes(set: &[KeyCode]) -> PressedCodes {
        let mut p = PressedCodes::default();
        for c in set {
            p.insert(*c);
        }
        p
    }

    /// The stuck-key cure. Whatever ate a release — a keyboard that vanished
    /// mid-press, a uinput write that failed, a device node replaced under us —
    /// the kernel still knows which keys are physically down. Anything we told
    /// apps was down that the kernel says is up must be released.
    #[test]
    fn reconcile_releases_keys_the_kernel_says_are_up() {
        let mut fwd = Forwarder::new(Recorder::default());
        fwd.forward(ev(KeyCode::KEY_W, true));
        fwd.forward(ev(KeyCode::KEY_S, true));
        fwd.sink.events.clear();

        // The kernel reports only W still held: S's release was lost.
        let stray = fwd.reconcile(&codes(&[KeyCode::KEY_W]));

        assert_eq!(stray, vec![KeyCode::KEY_S]);
        assert!(fwd.is_down(KeyCode::KEY_W), "a held key must stay held");
        assert!(!fwd.is_down(KeyCode::KEY_S));
        assert_eq!(fwd.sink.events, vec![(KeyCode::KEY_S.code(), 0)]);
    }

    #[test]
    fn reconcile_is_silent_when_nothing_drifted() {
        let mut fwd = Forwarder::new(Recorder::default());
        fwd.forward(ev(KeyCode::KEY_A, true));
        fwd.sink.events.clear();
        assert!(fwd.reconcile(&codes(&[KeyCode::KEY_A])).is_empty());
        assert!(fwd.sink.events.is_empty());
    }

    /// A stale key in the state machine's pressed set reads as an extra
    /// modifier, and `extra_mods` then blocks the hotkey forever — the
    /// "it just stops activating" half of the bug.
    #[test]
    fn sync_pressed_unblocks_a_hotkey_wedged_by_a_stale_key() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Toggle;
        // A Shift press whose release never arrived.
        s.handle(KeyCode::KEY_LEFTSHIFT, true, m);
        s.handle(KeyCode::KEY_LEFTCTRL, true, m);
        assert!(
            s.handle(KeyCode::KEY_SPACE, true, m).is_none(),
            "stale Shift should block the combo before the sweep"
        );
        s.handle(KeyCode::KEY_SPACE, false, m);

        // The kernel says only Ctrl is really down.
        assert!(
            s.sync_pressed(&codes(&[KeyCode::KEY_LEFTCTRL]), m)
                .is_none()
        );

        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, m),
            Some(HotkeyAction::Toggle)
        ));
    }

    /// A hold whose keys vanished must end the session, not leave the mic open.
    #[test]
    fn sync_pressed_stops_a_hold_whose_keys_vanished() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Hold;
        s.handle(KeyCode::KEY_LEFTCTRL, true, m);
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, m),
            Some(HotkeyAction::Start)
        ));
        assert!(matches!(
            s.sync_pressed(&PressedCodes::default(), m),
            Some(HotkeyAction::Stop)
        ));
        // Idempotent: a second sweep must not re-emit.
        assert!(s.sync_pressed(&PressedCodes::default(), m).is_none());
    }

    /// A withheld modifier press must never outlive the chord it belongs to.
    /// Before this, holding Ctrl and then reaching for the mouse (Ctrl+click,
    /// Ctrl+scroll) delivered no Ctrl at all — the press sat in `deferred`
    /// waiting for a key event that never came.
    #[test]
    fn an_idle_flush_delivers_a_withheld_modifier() {
        let m = ActivationMode::Hold;
        let combo = build_combo_mods(&[Modifier::Ctrl]);
        let mut state = ctrl_space_state();
        let mut fwd = Forwarder::new(Recorder::default());
        let mut consumed = PressedCodes::default();
        let mut deferred: Vec<InputEvent> = Vec::new();

        let ctrl = KeyCode::KEY_LEFTCTRL;
        process_grabbed_key(
            ev(ctrl, true),
            ctrl,
            true,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
            &combo,
            m,
        );
        assert!(!fwd.is_down(ctrl), "the press is deferred, as designed");

        // No further key event arrives; the idle tick resolves it.
        fwd.flush(&mut deferred);

        assert!(
            fwd.is_down(ctrl),
            "Ctrl must reach apps once the chord ends"
        );
        assert!(deferred.is_empty());
    }

    /// A sweep only knows what the probes tell it, and a probe that never
    /// opened — a keyboard whose udev ACL landed after the reader's own open —
    /// tells it nothing. Treating that silence as "the kernel says every key is
    /// up" would have the repair mechanism release keys the user is holding,
    /// which is the failure it exists to prevent. Without kernel truth the
    /// sweep must do nothing.
    #[test]
    fn a_sweep_without_kernel_truth_releases_nothing() {
        let mut fwd = Forwarder::new(Recorder::default());
        let mut state = ctrl_space_state();
        let mut consumed = PressedCodes::default();

        let shift = KeyCode::KEY_LEFTSHIFT;
        fwd.forward(ev(shift, true));
        fwd.sink.events.clear();

        // One watched keyboard, no probe handle for it.
        let probes = Probes {
            handles: &[],
            watched: 1,
        };
        sweep(
            &mut fwd,
            &mut state,
            &mut consumed,
            probes,
            ActivationMode::Hold,
        );

        assert!(
            fwd.is_down(shift),
            "a held key was released on a sweep with no kernel truth: {:?}",
            fwd.sink.events
        );
        assert!(fwd.sink.events.is_empty(), "{:?}", fwd.sink.events);
    }

    /// A gaming mouse's keyboard interface has a full keyboard key map and no
    /// pointer bits of its own, so capability alone cannot tell it apart from a
    /// real keyboard. Its sibling pointer node can. Grabbing one was the flaky
    /// half of this bug: a wireless mouse sleeps and re-enumerates on its own,
    /// and the grab dies with it.
    #[test]
    fn a_keyboard_node_sharing_an_id_with_a_pointer_is_not_grabbed() {
        let mouse = (0x1038u16, 0x1842u16);
        let keychron = (0x3434u16, 0x0151u16);
        let pointers = [mouse];

        assert!(!node_is_grabbable(true, mouse, &pointers));
        assert!(node_is_grabbable(true, keychron, &pointers));
        assert!(!node_is_grabbable(false, keychron, &pointers));
    }

    /// Rescans are rate-limited so a device that fails to open cannot spin the
    /// listener respawning a reader thread for it.
    #[test]
    fn rescan_waits_for_the_cooldown_even_after_a_reader_dies() {
        assert!(!rescan_due(RESCAN_COOLDOWN_TICKS - 1, true, true));
    }

    #[test]
    fn a_lost_reader_or_a_new_node_triggers_a_rescan() {
        // A keyboard that vanished and came back on the same node name: the
        // directory looks unchanged, so the dead reader is the only signal.
        assert!(rescan_due(RESCAN_COOLDOWN_TICKS, true, false));
        // A node appeared while every reader is still healthy.
        assert!(rescan_due(RESCAN_COOLDOWN_TICKS, false, true));
    }

    #[test]
    fn a_quiet_unchanged_system_is_not_rescanned() {
        assert!(!rescan_due(RESCAN_COOLDOWN_TICKS * 10, false, false));
    }

    #[test]
    fn release_all_releases_every_forwarded_down_key() {
        let mut fwd = Forwarder::new(Recorder::default());
        fwd.forward(ev(KeyCode::KEY_LEFTCTRL, true));
        fwd.forward(ev(KeyCode::KEY_A, true));
        fwd.forward(ev(KeyCode::KEY_A, false));
        assert!(fwd.is_down(KeyCode::KEY_LEFTCTRL));
        assert!(!fwd.is_down(KeyCode::KEY_A));

        fwd.release_all();

        assert!(!fwd.is_down(KeyCode::KEY_LEFTCTRL));
        assert!(
            fwd.sink
                .events
                .iter()
                .any(|&(c, v)| c == KeyCode::KEY_LEFTCTRL.code() && v == 0),
            "expected a synthetic Ctrl release"
        );
    }
}
