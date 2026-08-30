//! Focused-window capture on Windows.
//!
//! Where the Unix side has to negotiate with whichever compositor is running,
//! Windows answers in three calls: which window is in front, which process
//! owns it, and what that process's image is called. The executable stem is
//! the closest analogue to an X11 `WM_CLASS` / Wayland `app_id` — it is what
//! distinguishes `WindowsTerminal` from `firefox` — so that is what fills
//! `WindowTarget::class`.

use tracing::debug;

use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::Threading::{
    OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::{FocusToken, WindowTarget};

/// Snapshot the foreground window.
///
/// Every step degrades to `WindowTarget::unknown()` rather than failing: there
/// is legitimately no foreground window during a desktop switch or on the
/// secure desktop, and a process the app may not open (one running elevated,
/// or as another user) is normal, not exceptional.
pub(super) fn capture() -> WindowTarget {
    // SAFETY: takes no arguments and returns a handle or null.
    let hwnd = unsafe { GetForegroundWindow() };
    if hwnd.is_null() {
        debug!("no foreground window; delivery will paste wherever focus lands");
        return WindowTarget::unknown();
    }
    let token = Some(FocusToken::Win32Window(hwnd as isize));

    let Some(pid) = process_id(hwnd) else {
        return WindowTarget {
            focus_token: token,
            ..WindowTarget::unknown()
        };
    };
    let Some(stem) = image_stem(pid) else {
        return WindowTarget {
            focus_token: token,
            ..WindowTarget::unknown()
        };
    };

    debug!(class = %stem, "focused window detected");
    WindowTarget::from_class(stem, token)
}

/// The process owning a window, or `None` if the window has gone.
fn process_id(hwnd: windows_sys::Win32::Foundation::HWND) -> Option<u32> {
    let mut pid: u32 = 0;
    // SAFETY: `hwnd` came from `GetForegroundWindow`; `pid` is a live local.
    // A zero return means the window is invalid, which is why `pid` is checked
    // rather than trusted.
    unsafe { GetWindowThreadProcessId(hwnd, &mut pid) };
    (pid != 0).then_some(pid)
}

/// The executable's file stem for a process id — `WindowsTerminal`, `firefox`.
///
/// The extension is dropped here, not by the caller: `is_terminal_class`
/// splits on `.` and would otherwise reduce every Windows class to `exe`.
fn image_stem(pid: u32) -> Option<String> {
    // `PROCESS_QUERY_LIMITED_INFORMATION` is the least this needs, and unlike
    // the fuller query right it is granted for processes at a higher integrity
    // level — which is exactly when a dictation target is most likely to be a
    // console window.
    // SAFETY: a plain FFI call; a failure returns a null handle.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        debug!(pid, "cannot open foreground process; window class unknown");
        return None;
    }

    // Sized past `MAX_PATH`, not at it: `QueryFullProcessImageNameW` fails
    // outright with `ERROR_INSUFFICIENT_BUFFER` rather than truncating, and a
    // 260-char buffer turns any long-path install into an unknown window
    // class — which silently disables terminal detection. Heap, because this
    // is one allocation per dictation and 64 KB does not belong on the stack.
    let mut buf = vec![0u16; 32768];
    let mut len = buf.len() as u32;
    // SAFETY: `handle` is live until `CloseHandle` below; `buf`/`len` describe
    // a buffer the call writes at most `len` wide chars into, updating `len`
    // with how many it actually wrote.
    let ok = unsafe { QueryFullProcessImageNameW(handle, 0, buf.as_mut_ptr(), &mut len) };
    // SAFETY: `handle` came from `OpenProcess` and is closed exactly once.
    unsafe { CloseHandle(handle) };
    if ok == 0 {
        return None;
    }

    let path = String::from_utf16_lossy(&buf[..len as usize]);
    super::windows_image_stem(&path)
}
