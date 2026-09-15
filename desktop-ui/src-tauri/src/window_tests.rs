// SPDX-License-Identifier: GPL-3.0-or-later

//! Sizing rules for the main window's adaptive default.
//!
//! `adaptive_window_size` itself needs a live `AppHandle`, so the clamping it
//! delegates to is checked here directly.

use crate::{
    MAX_WINDOW_H, MAX_WINDOW_W, MIN_WINDOW_H, MIN_WINDOW_W, WINDOW_SCREEN_FRACTION, window_size_for,
};

#[test]
fn a_mid_size_monitor_gets_the_screen_fraction() {
    let (w, h) = window_size_for(1920.0, 1080.0);
    assert_eq!(w, 1920.0 * WINDOW_SCREEN_FRACTION);
    assert_eq!(h, 1080.0 * WINDOW_SCREEN_FRACTION);
}

#[test]
fn an_ultrawide_is_capped_not_sprawled() {
    // 3440x1440 at 70% would be 2408 wide — wider than any layout needs.
    let (w, h) = window_size_for(3440.0, 1440.0);
    assert_eq!(w, MAX_WINDOW_W);
    assert_eq!(h, 1440.0 * WINDOW_SCREEN_FRACTION);
}

#[test]
fn a_small_laptop_never_opens_below_the_designed_minimum() {
    let (w, h) = window_size_for(1366.0, 768.0);
    assert_eq!(w, MIN_WINDOW_W);
    assert_eq!(h, MIN_WINDOW_H);
}

#[test]
fn a_4k_monitor_is_capped_on_both_axes() {
    let (w, h) = window_size_for(3840.0, 2160.0);
    assert_eq!(w, MAX_WINDOW_W);
    assert_eq!(h, MAX_WINDOW_H);
}

#[test]
fn a_non_finite_logical_size_still_lands_inside_the_bounds() {
    // `f64::clamp` propagates NaN, so without the finite check this returns a
    // NaN size the window builder would be handed directly.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (w, h) = window_size_for(bad, bad);
        assert!((MIN_WINDOW_W..=MAX_WINDOW_W).contains(&w), "width was {w}");
        assert!((MIN_WINDOW_H..=MAX_WINDOW_H).contains(&h), "height was {h}");
    }
}
