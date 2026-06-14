pub mod calibration;
mod capture;
pub mod delivery;
pub mod events;
pub mod session;
#[cfg(unix)]
mod uinput_typer;

#[cfg(unix)]
mod cosmic_toplevel;
#[cfg(unix)]
mod wayland_toplevel;
#[cfg(unix)]
pub use uinput_typer::{WindowTarget, capture_focus_target};

pub use delivery::DeliveryService;
pub use session::{SessionManager, SessionManagerConfig};
