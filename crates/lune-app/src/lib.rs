mod capture;
pub mod delivery;
pub mod events;
pub mod session;
pub mod sound;
#[cfg(unix)]
mod uinput_typer;
// IBus input-method delivery backend — Linux only (Windows/macOS have their own
// delivery path in delivery.rs). See ibus.rs.
#[cfg(target_os = "linux")]
mod ibus;

#[cfg(unix)]
mod cosmic_toplevel;
#[cfg(unix)]
mod wayland_toplevel;
// XWayland-fallback decision for the pill overlay (Wayland-without-layer-shell).
#[cfg(target_os = "linux")]
mod compositor;
#[cfg(target_os = "linux")]
pub use compositor::should_use_xwayland_fallback;
#[cfg(unix)]
pub use uinput_typer::{WindowTarget, capture_focus_target};

pub use delivery::DeliveryService;
pub use session::{SessionManager, SessionManagerConfig};
