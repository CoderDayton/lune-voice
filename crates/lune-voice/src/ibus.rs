// SPDX-License-Identifier: GPL-3.0-or-later

//! IBus input-method delivery backend — **Linux only**.
//!
//! Windows and macOS deliver text through the `arboard` + `enigo` path in
//! `delivery.rs`; this module must never be compiled for them (it is gated
//! `#[cfg(target_os = "linux")]` in `lib.rs`, and its only dependency, `zbus`,
//! is gated to `cfg(target_os = "linux")` in `Cargo.toml`).
//!
//! ## Why
//!
//! The uinput backend pastes via `Ctrl+V` / `Ctrl+Shift+V` and has to detect
//! whether the focused window is a terminal to pick the chord. On GNOME Wayland
//! that detection is impossible (Mutter blocks `Shell.Eval`, ships no
//! `wlr-foreign-toplevel`, and XWayland tools can't see native windows), so the
//! paste silently no-ops in terminals. IBus sidesteps the whole problem: it
//! commits Unicode text straight into the focused **input context** through the
//! input-method protocol — no window detection, no paste chord, no terminal
//! list. This is the same mechanism Vocalinux uses to work on GNOME.
//!
//! ## How
//!
//! In-process IBus engine over `zbus` (blocking — the delivery thread is sync,
//! single-owner). At startup we connect to the IBus private bus, serve a
//! `Factory` + `Engine` object, and `RegisterComponent` so the daemon knows our
//! `lune-voice` engine. To inject we momentarily make our engine the global
//! engine, emit `CommitText`, then restore the user's previous engine — so the
//! user never configures IBus and normal typing keeps their layout.
//!
//! The IBus serialization formats below are taken verbatim from the IBus C
//! source (`src/ibusserializable.c`, `ibustext.c`, `ibusenginedesc.c`,
//! `ibuscomponent.c`): every serializable is a tuple `(s a{sv} …)` of the GType
//! name, an attachments dict, then type-specific fields.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use tracing::{debug, info, warn};
use zbus::blocking::{Connection, connection};
use zbus::zvariant::Value;

const ENGINE_NAME: &str = "lune-voice";
const COMPONENT_NAME: &str = "org.freedesktop.IBus.LuneVoice";
const ENGINE_PATH: &str = "/org/freedesktop/IBus/Engine/LuneVoice";
const FACTORY_PATH: &str = "/org/freedesktop/IBus/Factory";

const IBUS_SERVICE: &str = "org.freedesktop.IBus";
const IBUS_PATH: &str = "/org/freedesktop/IBus";
const IBUS_IFACE: &str = "org.freedesktop.IBus";
const ENGINE_IFACE: &str = "org.freedesktop.IBus.Engine";

/// How long to wait for IBus to focus/enable our engine after activation before
/// emitting `CommitText` (the commit-before-focus race). Bounded so a missed
/// callback never stalls delivery for long.
const FOCUS_WAIT: Duration = Duration::from_millis(200);

/// Longest a D-Bus call to the IBus daemon may block the delivery thread. A
/// wedged daemon otherwise stalls every later dictation behind one call.
const METHOD_TIMEOUT: Duration = Duration::from_secs(2);

// ─── Crash recovery ──────────────────────────────────────────────────────────

/// File recording the engine we displaced, written before we make ours global
/// and removed once the user's engine is back.
///
/// Restoring is otherwise `Drop`-only, and `Drop` does not run on `SIGTERM`
/// (logout, `systemctl stop`), on `SIGKILL`, or on a panic — the release profile
/// sets `panic = "abort"`. Without this file a death inside the ~200 ms commit
/// window leaves IBus pointed at our no-op engine, and since `try_new` refuses
/// to treat `lune-voice` as a restore target, *restarting the app would not fix
/// it either*: the user is left unable to type in any application.
const DISPLACED_ENGINE_FILE: &str = "displaced-input-engine";

fn displaced_engine_path() -> Option<std::path::PathBuf> {
    lune_persist::AppPaths::resolve().map(|p| p.data_dir().join(DISPLACED_ENGINE_FILE))
}

/// Record `name` as the engine to restore if this process dies before it can.
/// Best-effort: failing to write must never block a commit.
fn record_displaced_engine(name: &str) {
    if let Some(path) = displaced_engine_path() {
        record_displaced_engine_at(&path, name);
    }
}

fn record_displaced_engine_at(path: &std::path::Path, name: &str) {
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        debug!(error = %e, "ibus: could not create state dir for engine recovery");
        return;
    }
    if let Err(e) = std::fs::write(path, name) {
        debug!(error = %e, "ibus: could not record displaced engine");
    }
}

/// Drop the recovery record — the user's engine is back where it belongs.
fn clear_displaced_engine() {
    if let Some(path) = displaced_engine_path() {
        clear_displaced_engine_at(&path);
    }
}

fn clear_displaced_engine_at(path: &std::path::Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => debug!(error = %e, "ibus: could not clear displaced-engine record"),
    }
}

/// Read back a recorded engine name, if a previous run left one.
fn recorded_displaced_engine() -> Option<String> {
    recorded_displaced_engine_at(&displaced_engine_path()?)
}

fn recorded_displaced_engine_at(path: &std::path::Path) -> Option<String> {
    let name = std::fs::read_to_string(path).ok()?;
    let name = name.trim();
    // Never hand back our own engine as a restore target — that is the state we
    // are recovering *from*. A record naming it would send the next launch
    // straight back into the lockout it is trying to escape.
    if name.is_empty() || name == ENGINE_NAME {
        return None;
    }
    Some(name.to_owned())
}

// ─── Active-IM gate ──────────────────────────────────────────────────────────

/// True only when IBus is the active input method for this session, and the
/// user hasn't pinned the uinput path via `LUNE_DISABLE_IBUS`. Cheap (env only)
/// so non-IBus sessions pay no D-Bus cost — checked *before* `try_new`.
///
/// `GTK_IM_MODULE=ibus` / `QT_IM_MODULE=ibus` (or `XMODIFIERS=@im=ibus`) is the
/// authoritative signal that toolkits route input through IBus. A running
/// daemon alone is not sufficient: if the toolkits aren't pointed at IBus, a
/// commit would go nowhere, so we must fall through to uinput.
pub fn is_active_im() -> bool {
    if std::env::var_os("LUNE_DISABLE_IBUS").is_some() {
        debug!("ibus: disabled via LUNE_DISABLE_IBUS");
        return false;
    }
    let has_ibus = |key: &str| {
        std::env::var(key)
            .map(|v| v.to_ascii_lowercase().contains("ibus"))
            .unwrap_or(false)
    };
    has_ibus("GTK_IM_MODULE") || has_ibus("QT_IM_MODULE") || has_ibus("XMODIFIERS")
}

// ─── IBus private-bus address discovery ──────────────────────────────────────

/// Find the IBus private bus address: `$IBUS_ADDRESS`, then `ibus address`,
/// then the per-session address file under `~/.config/ibus/bus/`.
fn discover_ibus_address() -> Result<String, String> {
    if let Ok(addr) = std::env::var("IBUS_ADDRESS")
        && !addr.is_empty()
    {
        return Ok(addr);
    }

    // `ibus address` prints the bus address (or "(null)" when not running).
    if let Some(addr) = ibus_address_cmd() {
        return Ok(addr);
    }

    read_ibus_address_file()
}

/// Run `ibus address` with a hard 1s timeout. A wedged `ibus` binary must never
/// hang the delivery thread (mirrors the bounded-subprocess pattern in
/// `uinput_typer.rs`). Returns the address only on clean, non-empty success.
fn ibus_address_cmd() -> Option<String> {
    use std::process::{Command, Stdio};
    use std::time::Instant;

    let mut child = Command::new("ibus")
        .arg("address")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let out = child.wait_with_output().ok()?;
                let addr = String::from_utf8_lossy(&out.stdout).trim().to_string();
                return (!addr.is_empty() && addr != "(null)").then_some(addr);
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Parse `IBUS_ADDRESS=…` out of `~/.config/ibus/bus/<machine-id>-<display>`.
fn read_ibus_address_file() -> Result<String, String> {
    let config = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.config")))
        .ok_or("no XDG_CONFIG_HOME or HOME")?;
    let dir = std::path::Path::new(&config).join("ibus").join("bus");
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("read {dir:?}: {e}"))?;
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for line in text.lines() {
            if let Some(addr) = line.strip_prefix("IBUS_ADDRESS=") {
                let addr = addr.trim();
                if !addr.is_empty() {
                    return Ok(addr.to_string());
                }
            }
        }
    }
    Err("no IBus address file found".into())
}

// ─── Served D-Bus objects ────────────────────────────────────────────────────

/// Shared focus/enable state, set from the served `Engine` callbacks and waited
/// on by `commit_text` to avoid emitting before IBus has wired up our engine.
#[derive(Default)]
struct EngineState {
    enabled: Mutex<bool>,
    cv: Condvar,
}

impl EngineState {
    // Recover the guard on a poisoned lock rather than silently degrading — the
    // critical sections here never panic, so the inner state is always sound.
    fn set_active(&self, active: bool) {
        let mut g = self.enabled.lock().unwrap_or_else(|e| e.into_inner());
        *g = active;
        self.cv.notify_all();
    }
    /// Block until the engine is marked active or `timeout` elapses.
    /// Returns whether the engine actually became active — discarding this
    /// let a commit be emitted into a context IBus never focused/enabled,
    /// losing the dictated text with nothing logged.
    #[must_use]
    fn wait_active(&self, timeout: Duration) -> bool {
        let g = self.enabled.lock().unwrap_or_else(|e| e.into_inner());
        match self.cv.wait_timeout_while(g, timeout, |active| !*active) {
            Ok((_, result)) => !result.timed_out(),
            Err(poisoned) => !poisoned.into_inner().1.timed_out(),
        }
    }
    fn reset(&self) {
        let mut g = self.enabled.lock().unwrap_or_else(|e| e.into_inner());
        *g = false;
    }
}

/// `org.freedesktop.IBus.Factory` — the daemon calls `CreateEngine` when our
/// engine becomes global; we serve a single persistent engine object.
struct LuneFactory;

#[zbus::interface(name = "org.freedesktop.IBus.Factory")]
impl LuneFactory {
    fn create_engine(
        &self,
        _engine_name: &str,
    ) -> zbus::fdo::Result<zbus::zvariant::OwnedObjectPath> {
        zbus::zvariant::OwnedObjectPath::try_from(ENGINE_PATH)
            .map_err(|e| zbus::fdo::Error::Failed(format!("engine path: {e}")))
    }
}

/// `org.freedesktop.IBus.Engine` — a transparent proxy engine. It never
/// consumes real keystrokes (`process_key_event` returns `false`, so the user's
/// keyboard is untouched while we're briefly global); we only emit `CommitText`.
struct LuneEngine {
    state: Arc<EngineState>,
}

#[zbus::interface(name = "org.freedesktop.IBus.Engine")]
impl LuneEngine {
    /// Pass every key through unhandled — we never intercept input.
    fn process_key_event(&self, _keyval: u32, _keycode: u32, _state: u32) -> bool {
        false
    }
    fn focus_in(&self) {
        self.state.set_active(true);
    }
    fn focus_out(&self) {
        self.state.set_active(false);
    }
    // Newer IBus passes an object-path + client string; treat the same.
    fn focus_in_id(&self, _object_path: &str, _client: &str) {
        self.state.set_active(true);
    }
    fn focus_out_id(&self, _object_path: &str) {
        self.state.set_active(false);
    }
    fn enable(&self) {
        self.state.set_active(true);
    }
    fn disable(&self) {
        self.state.set_active(false);
    }
    fn reset(&self) {}
    fn set_capabilities(&self, _caps: u32) {}
    fn set_cursor_location(&self, _x: i32, _y: i32, _w: i32, _h: i32) {}
    fn set_surrounding_text(&self, _text: Value<'_>, _cursor: u32, _anchor: u32) {}
    fn property_activate(&self, _name: &str, _state: u32) {}
    fn property_show(&self, _name: &str) {}
    fn property_hide(&self, _name: &str) {}
    fn candidate_clicked(&self, _index: u32, _button: u32, _state: u32) {}
    fn page_up(&self) {}
    fn page_down(&self) {}
    fn cursor_up(&self) {}
    fn cursor_down(&self) {}
    fn destroy(&self) {}
}

// ─── IBus serialization (from the IBus C source) ─────────────────────────────

/// Owned string field.
fn s(v: &str) -> Value<'static> {
    Value::from(v.to_string())
}

/// Empty attachments dict `a{sv}` — the second field of every serializable.
fn attachments() -> Value<'static> {
    Value::from(HashMap::<String, Value>::new())
}

/// Empty array-of-variants `av`.
fn empty_av() -> Value<'static> {
    Value::from(Vec::<Value>::new())
}

/// Build a `(...)` structure from a flat list of fields. Used (rather than the
/// tuple `From` impls) because IBus serializables exceed the tuple arity —
/// `IBusEngineDesc` has 19 fields.
fn structure(fields: Vec<Value<'static>>) -> Value<'static> {
    let mut builder = zbus::zvariant::StructureBuilder::new();
    for field in fields {
        builder = builder.append_field(field);
    }
    Value::from(
        builder
            .build()
            .expect("ibus serializable: well-formed fields"),
    )
}

/// `IBusAttrList` → `(s a{sv} av)` = name, attachments, empty attributes.
fn ibus_attr_list() -> Value<'static> {
    structure(vec![s("IBusAttrList"), attachments(), empty_av()])
}

/// `IBusText` → `(s a{sv} s v)` = name, attachments, the text, a variant
/// wrapping an `IBusAttrList`. (`src/ibustext.c::ibus_text_serialize`.)
fn ibus_text(text: &str) -> Value<'static> {
    structure(vec![
        s("IBusText"),
        attachments(),
        s(text),
        Value::Value(Box::new(ibus_attr_list())),
    ])
}

/// `IBusEngineDesc` → `(s a{sv} s×8 u s×8)` in the field order from
/// `src/ibusenginedesc.c`: name, longname, description, language, license,
/// author, icon, layout, rank, hotkeys, symbol, setup, layout_variant,
/// layout_option, version, textdomain, icon_prop_key.
fn ibus_engine_desc() -> Value<'static> {
    structure(vec![
        s("IBusEngineDesc"),
        attachments(),
        s(ENGINE_NAME),              // name
        s("Lune Voice"),             // longname
        s("Lune Voice dictation"),   // description
        s("en"),                     // language
        s("GPL"),                    // license
        s("Lune Voice"),             // author
        s("audio-input-microphone"), // icon
        s("us"),                     // layout
        Value::from(99u32),          // rank (high so SetGlobalEngine resolves us)
        s(""),                       // hotkeys
        s(""),                       // symbol
        s(""),                       // setup
        s(""),                       // layout_variant
        s(""),                       // layout_option
        s("1.0"),                    // version
        s("lune-voice"),             // textdomain
        s(""),                       // icon_prop_key
    ])
}

/// `IBusComponent` → `(s a{sv} s×8 av av)`: name, description, version, license,
/// author, homepage, exec, textdomain, observed-paths (empty), engines (one
/// variant-wrapped `IBusEngineDesc`). (`src/ibuscomponent.c`.)
fn ibus_component() -> Value<'static> {
    // Each `Value` is already serialized as a `v` array element — no extra wrap.
    let engines = Value::from(vec![ibus_engine_desc()]);
    structure(vec![
        s("IBusComponent"),
        attachments(),
        s(COMPONENT_NAME),           // name
        s("Lune Voice IBus engine"), // description
        s("1.0"),                    // version
        s("GPL"),                    // license
        s("Lune Voice"),             // author
        s(""),                       // homepage
        s(""),                       // exec (empty — we register in-process)
        s("lune-voice"),             // textdomain
        empty_av(),                  // observed paths
        engines,                     // engines
    ])
}

// ─── The backend ─────────────────────────────────────────────────────────────

/// In-process IBus engine. Owned by the delivery thread; never `Send`/shared.
pub struct IbusTyper {
    conn: Connection,
    state: Arc<EngineState>,
    /// The global engine in effect before we first activated ours — the
    /// fallback restore target when a per-commit `GetGlobalEngine` fails, and
    /// the engine restored on shutdown (`Drop`). `None` when there was none.
    original_engine: Option<String>,
}

impl IbusTyper {
    /// Connect, serve our objects, register the component, and confirm the
    /// machinery works with a startup activate→restore handshake. `Err` on any
    /// failure so the caller falls back to the uinput path.
    pub fn try_new() -> Result<Self, String> {
        let state = Arc::new(EngineState::default());
        let conn = connect_and_register(&state)?;

        let mut typer = Self {
            conn,
            state,
            original_engine: None,
        };
        // Remember the engine in effect now so we can always restore it later.
        //
        // Finding *our* engine already global means a previous run died mid-commit
        // without restoring. Simply filtering it out would leave `original_engine`
        // at `None`, so nothing would ever put the user back — the lockout would
        // survive the restart. Fall back to what that run recorded.
        typer.original_engine = match typer.global_engine_name() {
            Some(name) if name != ENGINE_NAME => Some(name),
            Some(_) => {
                let recovered = recorded_displaced_engine();
                match &recovered {
                    Some(name) => warn!(
                        engine = %name,
                        "ibus: previous run left our engine active; recovering the user's engine"
                    ),
                    None => warn!(
                        "ibus: our engine is active with no recorded engine to restore; \
                         the user may need to reselect their input method"
                    ),
                }
                recovered
            }
            None => None,
        };

        // Pre-pay the CreateEngine/FocusIn round-trip and verify activation
        // works; restore immediately so the user's engine is unchanged at rest.
        //
        // Record first: the self-check below displaces the user's engine for the
        // whole `FOCUS_WAIT` window, and a death inside *that* window is exactly
        // as unrecoverable as one inside a commit.
        if let Some(orig) = &typer.original_engine {
            record_displaced_engine(orig);
        }
        typer
            .set_global_engine(ENGINE_NAME)
            .map_err(|e| format!("activation self-check failed: {e}"))?;
        let activated = typer.state.wait_active(FOCUS_WAIT);
        match typer.original_engine.clone() {
            // Only drop the record once the user's engine is actually back.
            // Clearing it after a failed restore would throw away the one thing
            // that can undo the lockout — including the record we just
            // recovered from a previous run.
            Some(orig) => {
                if let Err(e) = typer.set_global_engine(&orig) {
                    warn!(error = %e, "ibus: self-check restore failed; recovery record kept");
                } else {
                    clear_displaced_engine();
                }
            }
            // Nothing was displaced, so any record on disk is stale debris.
            None => clear_displaced_engine(),
        }

        // The self-check only means something if activation was observed. Fail
        // here (after restoring the user's engine) so delivery.rs falls back to
        // UinputTyper rather than trusting a non-functional IBus session.
        if !activated {
            return Err(format!(
                "activation self-check timed out after {}ms; IBus never enabled our engine",
                FOCUS_WAIT.as_millis()
            ));
        }

        info!("ibus: engine registered and activation verified");
        Ok(typer)
    }

    /// Commit Unicode `text` into the focused input context. Logs and returns
    /// on any D-Bus error so one failed commit never poisons the connection.
    ///
    /// Returns whether the text can have landed: the engine was enabled for a
    /// focused input context and the commit was sent. IBus enables the engine
    /// only for an app that has an input context, so `false` also means the
    /// focused app does not use IBus and the commit went nowhere.
    pub fn commit_text(&mut self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }
        // Restore target: the engine active right now, else the one captured at
        // startup — so we never leave the user stuck on our no-op engine.
        let restore_to = self
            .global_engine_name()
            .filter(|p| p != ENGINE_NAME)
            .or_else(|| self.original_engine.clone());

        // Persist the restore target *before* displacing anything, so a death
        // anywhere in the window below is recoverable on the next launch.
        if let Some(target) = &restore_to {
            record_displaced_engine(target);
        }

        self.state.reset();
        if let Err(e) = self.activate() {
            warn!(error = %e, "ibus: SetGlobalEngine failed; commit skipped");
            clear_displaced_engine();
            return false;
        }
        // Wait for IBus to focus/enable our engine so the commit lands. Emit
        // anyway on timeout (the engine may enable a moment later), but report
        // it — a silent stuck commit is otherwise invisible.
        let enabled = self.state.wait_active(FOCUS_WAIT);
        if !enabled {
            warn!(
                timeout_ms = FOCUS_WAIT.as_millis() as u64,
                "ibus: engine not activated before commit; text may be lost"
            );
        }

        // CommitText payload is `(v)` wrapping a serialized IBusText. A `Value`
        // signal arg is serialized as `v` automatically; don't double-wrap.
        let payload = ibus_text(text);
        let sent = match self.conn.emit_signal(
            None::<&str>,
            ENGINE_PATH,
            ENGINE_IFACE,
            "CommitText",
            &(payload,),
        ) {
            Ok(()) => true,
            Err(e) => {
                warn!(error = %e, "ibus: CommitText emit failed");
                false
            }
        };

        // Restore the user's previous engine (never leave ours active).
        match &restore_to {
            Some(target) => match self.set_global_engine(target) {
                Ok(()) => clear_displaced_engine(),
                // Leave the record in place: the next launch is now the only
                // thing that can put the user back.
                Err(e) => warn!(error = %e, "ibus: failed to restore previous engine"),
            },
            None => clear_displaced_engine(),
        }
        enabled && sent
    }

    /// Activate our engine, rebuilding a dead connection once before giving up.
    ///
    /// The connection is opened once and reused for the life of the process.
    /// An `ibus-daemon` that restarts — a session change, a crash, `ibus
    /// restart` — leaves it talking to a peer that is gone, and every later
    /// call fails. Without a rebuild, one daemon hiccup silently drops every
    /// remaining dictation until the app itself is restarted.
    fn activate(&mut self) -> zbus::Result<()> {
        let Err(first) = self.set_global_engine(ENGINE_NAME) else {
            return Ok(());
        };
        warn!(error = %first, "ibus: activation failed; rebuilding the bus connection");
        match connect_and_register(&self.state) {
            Ok(conn) => {
                self.conn = conn;
                self.set_global_engine(ENGINE_NAME)
            }
            Err(e) => {
                warn!(error = %e, "ibus: reconnect failed");
                Err(first)
            }
        }
    }

    fn set_global_engine(&self, name: &str) -> zbus::Result<()> {
        self.conn
            .call_method(
                Some(IBUS_SERVICE),
                IBUS_PATH,
                Some(IBUS_IFACE),
                "SetGlobalEngine",
                &(name,),
            )
            .map(|_| ())
    }

    /// Name of the current global engine, if any. `GetGlobalEngine` returns a
    /// variant-wrapped `IBusEngineDesc`; its 3rd field (after name + attachments)
    /// is the engine name.
    fn global_engine_name(&self) -> Option<String> {
        let reply = self
            .conn
            .call_method(
                Some(IBUS_SERVICE),
                IBUS_PATH,
                Some(IBUS_IFACE),
                "GetGlobalEngine",
                &(),
            )
            .ok()?;
        let body = reply.body();
        let desc: Value = body.deserialize().ok()?;
        engine_name_from_desc(&desc)
    }
}

/// Open the IBus private bus, serve our factory and engine on it, and register
/// the component so `SetGlobalEngine("lune-voice")` resolves to our
/// in-process factory.
fn connect_and_register(state: &Arc<EngineState>) -> Result<Connection, String> {
    let address = discover_ibus_address()?;
    let conn = connection::Builder::address(address.as_str())
        .map_err(|e| format!("ibus address parse: {e}"))?
        .method_timeout(METHOD_TIMEOUT)
        .serve_at(FACTORY_PATH, LuneFactory)
        .map_err(|e| format!("serve factory: {e}"))?
        .serve_at(
            ENGINE_PATH,
            LuneEngine {
                state: Arc::clone(state),
            },
        )
        .map_err(|e| format!("serve engine: {e}"))?
        .build()
        .map_err(|e| format!("connect ibus bus: {e}"))?;

    conn.call_method(
        Some(IBUS_SERVICE),
        IBUS_PATH,
        Some(IBUS_IFACE),
        "RegisterComponent",
        // A `Value` arg is serialized as `v` automatically; don't double-wrap.
        &(ibus_component(),),
    )
    .map_err(|e| format!("RegisterComponent: {e}"))?;

    Ok(conn)
}

/// Restore the displaced engine now, outside the normal `Drop` path.
///
/// For the shutdown-signal handler: `Drop` never runs on `SIGTERM`, and the
/// typer itself lives on the delivery thread where a signal task cannot reach
/// it. Opening a fresh connection is the cost of not stranding the user until
/// their next launch. Returns whether something needed restoring *and* the
/// restore succeeded.
pub fn restore_displaced_engine_now() -> bool {
    let Some(target) = recorded_displaced_engine() else {
        return false;
    };
    let Ok(address) = discover_ibus_address() else {
        return false;
    };
    let Ok(conn) = connection::Builder::address(address.as_str())
        .and_then(|b| b.method_timeout(METHOD_TIMEOUT).build())
    else {
        return false;
    };
    let restored = conn
        .call_method(
            Some(IBUS_SERVICE),
            IBUS_PATH,
            Some(IBUS_IFACE),
            "SetGlobalEngine",
            &(target.as_str(),),
        )
        .is_ok();
    if restored {
        info!(engine = %target, "ibus: restored the user's engine on shutdown");
        clear_displaced_engine();
    }
    restored
}

impl Drop for IbusTyper {
    /// On shutdown, restore the engine we displaced so we never strand the user
    /// on our no-op engine.
    fn drop(&mut self) {
        if let Some(orig) = self.original_engine.clone()
            && self.set_global_engine(&orig).is_err()
        {
            // Keep the recovery record so the next launch can finish the job.
            return;
        }
        clear_displaced_engine();
    }
}

/// Extract the `name` field (index 2) from a serialized `IBusEngineDesc` value.
/// Variant nesting is depth-bounded so a malformed daemon reply can't blow the
/// stack.
fn engine_name_from_desc(desc: &Value<'_>) -> Option<String> {
    fn inner(desc: &Value<'_>, depth: u8) -> Option<String> {
        if depth > 4 {
            return None;
        }
        match desc {
            Value::Structure(s) => match s.fields().get(2)? {
                Value::Str(s) => Some(s.as_str().to_string()),
                _ => None,
            },
            Value::Value(boxed) => inner(boxed, depth + 1),
            _ => None,
        }
    }
    inner(desc, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the record: a run that dies mid-commit must leave
    /// enough behind for the next launch to put the user's engine back.
    #[test]
    fn a_recorded_engine_survives_for_the_next_launch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state").join(DISPLACED_ENGINE_FILE);

        record_displaced_engine_at(&path, "xkb:us::eng");
        assert_eq!(
            recorded_displaced_engine_at(&path).as_deref(),
            Some("xkb:us::eng")
        );

        clear_displaced_engine_at(&path);
        assert_eq!(recorded_displaced_engine_at(&path), None);
    }

    /// Recovering *to* our own engine would leave the user exactly as stuck as
    /// before, so a record naming it must read back as "nothing to restore".
    #[test]
    fn our_own_engine_is_never_a_restore_target() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(DISPLACED_ENGINE_FILE);

        record_displaced_engine_at(&path, ENGINE_NAME);
        assert_eq!(recorded_displaced_engine_at(&path), None);
    }

    #[test]
    fn a_blank_or_missing_record_yields_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join(DISPLACED_ENGINE_FILE);
        assert_eq!(recorded_displaced_engine_at(&missing), None);

        record_displaced_engine_at(&missing, "   \n ");
        assert_eq!(recorded_displaced_engine_at(&missing), None);
    }

    /// Engine names round-trip verbatim apart from the trailing newline an
    /// editor or `echo` would add.
    #[test]
    fn a_recorded_name_is_trimmed_not_mangled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(DISPLACED_ENGINE_FILE);

        record_displaced_engine_at(&path, "anthy\n");
        assert_eq!(
            recorded_displaced_engine_at(&path).as_deref(),
            Some("anthy")
        );
    }

    /// Clearing a record that was never written is the common case (most
    /// commits restore cleanly), so it must not be an error path.
    #[test]
    fn clearing_a_missing_record_is_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        clear_displaced_engine_at(&dir.path().join(DISPLACED_ENGINE_FILE));
    }

    #[test]
    fn ibus_text_has_expected_signature() {
        // The exact IBusText wire shape IBus expects: name, attachments, text,
        // and a variant-wrapped attr list. A wrong signature makes IBus silently
        // drop the commit, so pin it.
        let v = ibus_text("hi");
        assert_eq!(v.value_signature().to_string(), "(sa{sv}sv)");
    }

    #[test]
    fn ibus_attr_list_signature() {
        let v = ibus_attr_list();
        assert_eq!(v.value_signature().to_string(), "(sa{sv}av)");
    }

    #[test]
    fn ibus_engine_desc_signature() {
        // (s a{sv} s×8 u s×8) — 17 typed fields after the header.
        let sig = ibus_engine_desc().value_signature().to_string();
        assert!(sig.starts_with("(sa{sv}"), "got {sig}");
        assert!(sig.contains('u'), "rank field missing: {sig}");
    }

    #[test]
    fn is_active_im_runs() {
        let _ = is_active_im();
    }

    /// Live validation against a running IBus daemon: registration + the
    /// activate→restore handshake must succeed (this exercises the IBusComponent
    /// serialization, the served Factory/Engine objects, and SetGlobalEngine).
    /// It does NOT commit any text, so it won't inject into a focused window.
    /// Run with: `cargo test -p lune-voice --lib ibus::tests::live -- --ignored --nocapture`.
    #[test]
    #[ignore = "requires a live IBus daemon"]
    fn live_register_and_restore() {
        if !is_active_im() {
            eprintln!("skip: IBus is not the active input method here");
            return;
        }
        let typer = IbusTyper::try_new().expect("try_new should register against live IBus");
        let after = typer.global_engine_name();
        println!("global engine after handshake: {after:?}");
        assert_ne!(
            after.as_deref(),
            Some(ENGINE_NAME),
            "handshake must restore the previous engine, not leave us global"
        );
    }
}
