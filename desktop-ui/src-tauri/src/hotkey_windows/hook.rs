//! The Win32 half of the Windows hotkey backend: a `WH_KEYBOARD_LL` hook and
//! the thread that pumps messages for it.
//!
//! Everything that decides *what a key means* lives in the parent module and
//! is portable. This file only moves bytes between Win32 and that state
//! machine, so it holds no policy of its own.
//!
//! Two constraints shape it:
//!
//! * A low-level hook callback runs on the thread that installed it, and only
//!   while that thread pumps a message loop. Hence a dedicated thread whose
//!   whole job is `GetMessageW`.
//! * Windows silently removes a hook whose callback overruns
//!   `LowLevelHooksTimeout` (~300 ms by default), and the hook sits in the
//!   delivery path of every keystroke on the machine. So the callback does no
//!   I/O, takes no lock that another thread holds while doing I/O, and
//!   publishes through `HotkeySink`, which cannot block.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use tracing::{info, warn};

use windows_sys::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, KBDLLHOOKSTRUCT, MSG, SetWindowsHookExW,
    TranslateMessage, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

use lune_core::config::{ActivationMode, HotkeyConfig};

use super::{Vk, WinHotkeyState, WinSnapshot};
use crate::hotkey::{HotkeyBackend, HotkeySink};

/// `nCode` value that means "this is a real event, act on it". Any negative
/// code obliges us to pass the event on untouched.
const HC_ACTION: i32 = 0;

/// `KBDLLHOOKSTRUCT::flags` bit marking a synthesised keystroke.
///
/// Load-bearing: this app types its own output with `enigo`, including the
/// Ctrl+V paste chord. Feeding those back into the state machine would let a
/// dictation's own delivery look like the user pressing modifiers.
const LLKHF_INJECTED: u32 = 0x10;

/// Swallow the key: return non-zero instead of chaining.
const SWALLOW: LRESULT = 1;

// ─── Live configuration ──────────────────────────────────────────────────────

/// The binding, swappable while the hook is installed.
///
/// Same shape as the Linux backend's `LiveHotkeyConfig` and for the same
/// reason: the hook thread reads `generation` once per key event (one atomic
/// load) and re-reads the snapshot only when it changed, so a settings save
/// never has to tear down the hook.
pub(super) struct LiveWinConfig {
    inner: RwLock<WinSnapshot>,
    generation: AtomicU64,
}

impl LiveWinConfig {
    fn new(snap: WinSnapshot) -> Self {
        Self {
            inner: RwLock::new(snap),
            generation: AtomicU64::new(0),
        }
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn snapshot(&self) -> WinSnapshot {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl HotkeyBackend for LiveWinConfig {
    fn update(&self, config: &HotkeyConfig, mode: ActivationMode) -> Result<(), String> {
        let snap = WinSnapshot::from_config(config, mode)
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

    fn name(&self) -> &'static str {
        "win-hook"
    }
}

// ─── Hook thread state ───────────────────────────────────────────────────────

/// Everything the callback touches. Thread-local because a low-level hook
/// callback always runs on the thread that installed it — so this is reachable
/// from exactly one thread and needs no lock of its own.
struct HookState {
    machine: WinHotkeyState,
    sink: HotkeySink,
    live: Arc<LiveWinConfig>,
    generation: u64,
}

thread_local! {
    static STATE: RefCell<Option<HookState>> = const { RefCell::new(None) };
}

/// Translate a Win32 message id into a key transition.
///
/// `WM_SYSKEYDOWN`/`WM_SYSKEYUP` carry Alt-modified keys and Alt itself; they
/// are ordinary transitions here, and omitting them would make every Alt
/// binding silently dead.
const fn transition(wparam: WPARAM) -> Option<bool> {
    match wparam as u32 {
        WM_KEYDOWN | WM_SYSKEYDOWN => Some(true),
        WM_KEYUP | WM_SYSKEYUP => Some(false),
        _ => None,
    }
}

/// The hook callback. Runs for every keystroke on the machine.
///
/// # Safety
/// Called by Windows with `lparam` pointing to a valid `KBDLLHOOKSTRUCT` for
/// the lifetime of the call whenever `code == HC_ACTION`.
unsafe extern "system" fn keyboard_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // A negative code is a documented "not yours to inspect" — chain it on
    // before touching anything else.
    if code < HC_ACTION {
        return unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) };
    }

    let swallow = STATE.with(|state| {
        let Ok(mut state) = state.try_borrow_mut() else {
            // Re-entered while borrowed. Impossible on one thread that never
            // re-enters, but a panic here would take the hook down and with it
            // every hotkey, so the safe answer is to pass the key through.
            return false;
        };
        let Some(state) = state.as_mut() else {
            return false;
        };

        let Some(pressed) = transition(wparam) else {
            return false;
        };
        // SAFETY: `code >= HC_ACTION` guarantees `lparam` is a live
        // `KBDLLHOOKSTRUCT` for the duration of this call.
        let event = unsafe { &*(lparam as *const KBDLLHOOKSTRUCT) };
        if event.flags & LLKHF_INJECTED != 0 {
            return false;
        }
        let Ok(vk) = Vk::try_from(event.vkCode) else {
            return false;
        };

        // Pick up a settings save without restarting the hook.
        let generation = state.live.generation();
        if generation != state.generation {
            state.machine.retarget(state.live.snapshot());
            state.generation = generation;
        }

        let decision = state.machine.handle(vk, pressed);
        if let Some(action) = decision.action {
            state.sink.emit(action);
        }
        decision.swallow
    });

    if swallow {
        return SWALLOW;
    }
    unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Install the hook on a dedicated thread and return the live binding.
///
/// `None` if the binding cannot be resolved, the thread will not start, or
/// `SetWindowsHookExW` refuses — the caller reads that as "no native backend"
/// and falls back to the global-shortcut plugin, so reporting success before
/// the hook is actually installed would leave the user with no hotkeys at all
/// and no fallback. Hence the wait: the answer arrives in microseconds, and
/// the thread then runs for the process lifetime, so its join handle is
/// dropped deliberately.
pub(super) fn start(
    config: &HotkeyConfig,
    mode: ActivationMode,
    sink: HotkeySink,
) -> Option<Arc<dyn HotkeyBackend>> {
    let snap = WinSnapshot::from_config(config, mode)?;
    let live = Arc::new(LiveWinConfig::new(snap.clone()));
    let thread_live = Arc::clone(&live);
    let (installed_tx, installed_rx) = std::sync::mpsc::channel();

    std::thread::Builder::new()
        .name("hotkey-winhook".into())
        .spawn(move || run_hook(snap, sink, thread_live, &installed_tx))
        .map_err(|e| warn!(error = %e, "failed to spawn hotkey hook thread"))
        .ok()?;

    // `Err` means the thread died before reporting, which is a failure too.
    if installed_rx.recv().unwrap_or(false) {
        Some(live)
    } else {
        None
    }
}

/// Install the hook, then pump messages until the process ends.
///
/// `installed` carries the one fact `start` cannot observe from outside the
/// thread, and is reported exactly once either way.
fn run_hook(
    snap: WinSnapshot,
    sink: HotkeySink,
    live: Arc<LiveWinConfig>,
    installed: &std::sync::mpsc::Sender<bool>,
) {
    let generation = live.generation();
    STATE.with(|state| {
        *state.borrow_mut() = Some(HookState {
            machine: WinHotkeyState::new(snap),
            sink,
            live,
            generation,
        });
    });

    // SAFETY: a null module handle with a zero thread id installs a global
    // low-level hook owned by this thread, which is what the callback's
    // thread-local state assumes.
    let handle =
        unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_proc), std::ptr::null_mut(), 0) };
    if handle.is_null() {
        warn!("SetWindowsHookExW failed; falling back to the global-shortcut plugin");
        let _ = installed.send(false);
        return;
    }
    let _ = installed.send(true);
    info!("low-level keyboard hook installed");

    // The hook only fires while this thread pumps messages. Nothing posts to
    // this thread, so the loop exists purely to keep the callback alive.
    let mut msg = MSG {
        hwnd: std::ptr::null_mut(),
        message: 0,
        wParam: 0,
        lParam: 0,
        time: 0,
        pt: windows_sys::Win32::Foundation::POINT { x: 0, y: 0 },
    };
    // SAFETY: `msg` is a valid, owned message buffer for the whole loop.
    while unsafe { GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) } > 0 {
        unsafe {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    warn!("hotkey hook message loop exited; global hotkeys are dead");
}
