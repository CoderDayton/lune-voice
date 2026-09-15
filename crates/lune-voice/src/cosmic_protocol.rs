// SPDX-License-Identifier: GPL-3.0-or-later

//! Client bindings for the COSMIC toplevel-info protocol, generated here from
//! vendored protocol XML.
//!
//! We generate these ourselves rather than depending on the `cosmic-protocols`
//! crate, which declares `GPL-3.0-only`. Lune Voice is GPL-3.0-**or-later**, so
//! linking a v3-only crate would pin the combined work to v3 and quietly defeat
//! the "or later". The protocol definitions that crate is generated from carry
//! no such restriction — both XML files under `protocols/` ship an X11-style
//! permissive grant from their authors. Running `wayland-scanner` over that XML
//! ourselves is exactly what `cosmic-protocols` does, and it keeps the upgrade
//! path open while dropping a dependency.
//!
//! This is not replaceable by `wayland-protocols`' standardized
//! `ext_foreign_toplevel_list_v1`: that protocol is deliberately minimal and
//! carries no toplevel *state*, so it can enumerate windows but cannot say
//! which one is activated. Focus detection needs the COSMIC protocol, because
//! cosmic-comp implements no wlr-foreign-toplevel equivalent either.
//!
//! Two XMLs are vendored because toplevel-info's deprecated v1
//! `workspace_enter` / `workspace_leave` events reference
//! `zcosmic_workspace_handle_v1`. We never read those events, but Wayland
//! opcodes are positional, so the definitions have to stay complete for the
//! rest of the protocol to decode correctly.

#![allow(
    dead_code,
    non_camel_case_types,
    unused_unsafe,
    unused_variables,
    non_upper_case_globals,
    non_snake_case,
    unused_imports,
    missing_docs,
    clippy::all
)]

/// `zcosmic_workspace_*`. Present only to satisfy toplevel-info's references
/// to it; nothing in the app drives workspaces.
pub mod workspace {
    pub mod client {
        use wayland_client;
        use wayland_client::protocol::*;

        pub mod __interfaces {
            use wayland_client::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("./protocols/cosmic-workspace-unstable-v1.xml");
        }
        use self::__interfaces::*;

        wayland_scanner::generate_client_code!("./protocols/cosmic-workspace-unstable-v1.xml");
    }
}

/// `zcosmic_toplevel_info_v1` — the protocol that actually reports which
/// toplevel is activated. Paths resolve relative to `CARGO_MANIFEST_DIR`.
pub mod toplevel_info {
    pub mod client {
        use wayland_client;
        use wayland_client::protocol::*;
        use wayland_protocols::ext::foreign_toplevel_list::v1::client::*;
        use wayland_protocols::ext::workspace::v1::client::*;

        use crate::cosmic_protocol::workspace::client::*;

        pub mod __interfaces {
            use wayland_client::protocol::__interfaces::*;
            use wayland_protocols::ext::foreign_toplevel_list::v1::client::__interfaces::*;
            use wayland_protocols::ext::workspace::v1::client::__interfaces::*;

            use crate::cosmic_protocol::workspace::client::__interfaces::*;

            wayland_scanner::generate_interfaces!(
                "./protocols/cosmic-toplevel-info-unstable-v1.xml"
            );
        }
        use self::__interfaces::*;

        wayland_scanner::generate_client_code!("./protocols/cosmic-toplevel-info-unstable-v1.xml");
    }
}
