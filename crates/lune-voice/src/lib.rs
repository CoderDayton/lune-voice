mod capture;
pub mod delivery;
mod redact;
mod sanitize;
pub mod session;
pub mod sound;
#[cfg(unix)]
mod uinput_typer;
// IBus input-method delivery backend — Linux only (Windows/macOS have their own
// delivery path in delivery.rs). See ibus.rs.
#[cfg(target_os = "linux")]
mod ibus;
#[cfg(target_os = "linux")]
pub use ibus::restore_displaced_engine_now;

// Bounded display-server probing, shared by the compositor and toplevel
// queries.
#[cfg(unix)]
mod cosmic_protocol;
#[cfg(unix)]
mod cosmic_toplevel;
#[cfg(unix)]
mod probe;
#[cfg(unix)]
mod toplevel;
#[cfg(unix)]
mod wayland_toplevel;
// XWayland-fallback decision for the pill overlay (Wayland-without-layer-shell).
#[cfg(target_os = "linux")]
mod compositor;
#[cfg(target_os = "linux")]
pub use compositor::should_use_xwayland_fallback;
#[cfg(unix)]
pub use uinput_typer::{
    grant_uinput_access, remove_uinput_rule, uinput_access_ready, uinput_rule_is_self_installed,
};

// Focused-window capture. Available on every platform: the type and its
// terminal classification are shared, and only the capture behind them is
// platform-specific.
mod focus;
pub use focus::{FocusToken, WindowTarget, capture_focus_target};

pub use delivery::{DeliveryHealth, DeliveryService};
pub use session::{SessionManager, SessionManagerConfig};
