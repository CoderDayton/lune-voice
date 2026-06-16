pub mod calibration;
mod capture;
pub mod delivery;
pub mod events;
mod gate;
pub mod session;
#[cfg(unix)]
mod uinput_typer;
// IBus input-method delivery backend — Linux only (Windows/macOS have their own
// delivery path in delivery.rs). See ibus.rs.
#[cfg(target_os = "linux")]
mod ibus;
pub mod vad;

#[cfg(unix)]
mod cosmic_toplevel;
#[cfg(unix)]
mod wayland_toplevel;
#[cfg(unix)]
pub use uinput_typer::{WindowTarget, capture_focus_target};

pub use delivery::DeliveryService;
pub use session::{SessionManager, SessionManagerConfig};
