#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// LUNE FORK: bindgen emits `transmute` for signed/unsigned bitfield accessors,
// which newer rustc flags. Generated code — nothing to fix here until bindgen
// stops.
#![allow(unnecessary_transmutes)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
