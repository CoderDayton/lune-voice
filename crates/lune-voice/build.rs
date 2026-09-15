// SPDX-License-Identifier: GPL-3.0-or-later

//! `src/cosmic_protocol.rs` expands the vendored COSMIC protocol XML through
//! `wayland-scanner`'s proc macros. Cargo cannot see files a proc macro reads,
//! so without these directives an edit to the XML would leave the generated
//! bindings stale until an unrelated change forced a rebuild.

/// Protocol definitions consumed by `wayland_scanner::generate_client_code!`,
/// relative to this crate's manifest directory.
const PROTOCOL_XML: [&str; 2] = [
    "protocols/cosmic-toplevel-info-unstable-v1.xml",
    "protocols/cosmic-workspace-unstable-v1.xml",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for xml in PROTOCOL_XML {
        println!("cargo:rerun-if-changed={xml}");
    }
}
